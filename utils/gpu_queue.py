"""Agende experimentos em processos isolados com controle estimado de VRAM.

Cada job é a tupla ``(dataset, regime, representation, model_name, cfg, seed)``
interpretada por ``main_family.execute_job``. O contexto multiprocessing
``spawn`` cria um processo por tentativa. Sua saída libera as alocações CUDA;
uma nova tentativa reinicia a seed, sem checkpoint de época.

O agendador limita a concorrência global, consulta ``nvidia-smi`` e respeita
índices ou UUIDs completos em ``CUDA_VISIBLE_DEVICES``. As reservas são locais
ao agendador e conservadoras; não constituem limite de memória do processo nem
garantia de que um modelo caberá. CPU requer seleção explícita.

Cada execução cria um diretório novo com ``status.json`` (lista de registros),
``<id>.log`` (saída acumulada das tentativas) e ``<id>.json`` (resultado da
última tentativa que conseguiu gravá-lo). Estados usuais são pending, running,
completed, failed e interrupted. OOM é um resultado intermediário do worker,
convertido em pending para repetir ou failed ao esgotar as tentativas.

O timeout de pendência inclui a espera pelo limite de concorrência desde a
criação da fila; não limita a duração de um job em execução. ``float('inf')``
permite esperar indefinidamente. Não há restauração do status de filas antigas:
resultados concluídos são reaproveitados pelo executor via ``metrics.json``.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import traceback


def gpu_memory():
    """Consulte a memória das GPUs permitidas pelo ambiente do agendador.

    Returns
    -------
    list[tuple[str, int, int]]
        Tuplas ``(uuid, memória_livre, memória_total)`` em MiB, na ordem
        retornada por ``nvidia-smi``. A lista pode estar vazia.

    Raises
    ------
    FileNotFoundError
        Se ``nvidia-smi`` não estiver disponível no PATH.
    subprocess.CalledProcessError
        Se a consulta terminar com código diferente de zero.
    ValueError
        Se a saída não tiver as três colunas ou valores inteiros esperados.

    Notes
    -----
    ``CUDA_VISIBLE_DEVICES`` ausente permite todas as linhas; caso presente,
    compara tokens separados por vírgulas com o índice da linha (base zero)
    ou seu UUID completo. Não normaliza espaços, resolve prefixos de UUID
    nem interpreta dispositivos MIG. Não modifica o ambiente ou reserva VRAM.
    """
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
        "--format=csv,noheader,nounits"], text=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None if visible is None else visible.split(",")
    rows = []
    for index, line in enumerate(output.strip().splitlines()):
        uuid, free, total = [part.strip() for part in line.split(",")]
        if allowed is None or str(index) in allowed or uuid in allowed:
            rows.append((uuid, int(free), int(total)))
    return rows


def worker(job, gpu, status, log):
    """Execute uma tentativa e grave seu resultado no processo filho.

    Parameters
    ----------
    job : tuple
        ``(dataset, regime, representation, model_name, cfg, seed)`` enviado
        sem alteração para ``main_family.execute_job``.
    gpu : str or None
        UUID da GPU a expor em CUDA_VISIBLE_DEVICES; None oculta as GPUs
        para execução em CPU.
    status : str or os.PathLike
        Arquivo JSON de resultado da tentativa, sobrescrito ao terminar.
    log : str or os.PathLike
        Arquivo aberto em append para stdout/stderr, com buffering por linha.

    Returns
    -------
    None
        O resultado é comunicado pelo arquivo, não pelo retorno do processo.

    Notes
    -----
    Limita as threads intra-op do PyTorch a uma. Verifica disponibilidade CUDA
    quando recebeu uma GPU, sem fallback silencioso para CPU. Sucesso grava
    apenas state=completed; ``torch.cuda.OutOfMemoryError`` grava state=oom;
    outras exceções capturadas gravam state=failed. Falhas incluem ``error``
    e ``traceback``, com o traceback também no log.

    ``completed`` inclui tanto treinamento finalizado quanto resultado antigo
    pulado pelo executor. Alterações de ambiente e streams duram até o fim do
    processo. BaseException, falhas ao abrir/gravar arquivos e problemas no
    próprio tratamento de erro podem impedir a escrita do status; o agendador
    trata a ausência do arquivo após a saída como falha do worker.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
    with open(log, "a", buffering=1) as stream:
        import sys
        sys.stdout = sys.stderr = stream
        try:
            import torch
            torch.set_num_threads(1)
            if gpu is not None and not torch.cuda.is_available():
                raise RuntimeError("GPU selecionada, mas CUDA indisponível no PyTorch")
            from main_family import execute_job
            execute_job(job)
            result = {"state": "completed"}
        except Exception as exc:
            import torch
            result = {"state": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                      "error": repr(exc), "traceback": traceback.format_exc()}
            traceback.print_exc()
        Path(status).write_text(json.dumps(result))


def schedule(jobs, args):
    """Processe a fila até não restarem jobs pendentes ou em execução.

    Parameters
    ----------
    jobs : iterable[tuple]
        Entradas ``(dataset, regime, representation, model_name, cfg, seed)``.
        São materializadas em registros com IDs sequenciais a partir de zero;
        seu conteúdo deve ser serializável em JSON e pelo multiprocessing.
    args : argparse.Namespace
        Opções esperadas, normalmente validadas por ``main_family.main``:

        * run_dir: caminho novo ou None para queue_runs/<time.time_ns()>.
        * device: ``cuda`` para consultar GPUs ou ``cpu`` para dispensá-las.
        * max_parallel: máximo global de processos ativos, não por GPU.
        * gpu_memory_mb: estimativa inicial em MiB por processo.
        * gpu_reserve_mb: margem livre adicional em MiB por GPU.
        * poll_seconds: intervalo entre ciclos, limitado a 30 segundos.
        * wait_timeout: segundos permitidos no estado pending; inf desativa.
        * oom_retries: novas tentativas permitidas além da tentativa inicial.

    Returns
    -------
    list[dict]
        Registros cujo estado final difere de completed; lista vazia indica
        sucesso de todos os jobs. Cada registro contém id, job, state,
        attempts, memory_mb e since; gpu, error e traceback são adicionados
        conforme as transições. ``since`` usa relógio monotônico, não data
        civil; ``attempts`` conta processos iniciados.

    Raises
    ------
    FileExistsError
        Se o diretório de relatório escolhido já existir.
    BaseException
        Erros no agendador e interrupções são propagados. Dentro do laço
        protegido, os filhos ativos são terminados e aguardados, e os jobs
        pending/running são marcados interrupted antes de repropagar, desde
        que a limpeza e a gravação do relatório tenham sucesso.

    Notes
    -----
    Percorre pendentes na ordem original, mas pode ultrapassar um job que
    não cabe em memória. Escolhe a GPU com maior memória livre após subtrair
    estimativas dos processos ativos nessa GPU e exige ainda a reserva.
    A consulta é feita uma vez por ciclo; descontar estimativas da memória
    livre medida pode contar novamente memória já alocada. Não há coordenação
    com reservas de outros agendadores, nem limitação física de VRAM.

    Ao sair um worker, lê seu JSON ou registra falha se ele não existir.
    OOM duplica a estimativa, reinicia o relógio de espera e recoloca o job
    na fila enquanto houver retries; com oom_retries=2 há até três tentativas.
    Outros erros do worker não são repetidos e não interrompem a fila.

    O relógio inicial é criado para todos os jobs ao montar os registros.
    O timeout é verificado antes da disponibilidade de vaga: portanto também
    expira jobs que só aguardavam outros treinamentos. Não cancela workers
    ativos. Expirações são registradas no JSON sem mensagem individual no
    stdout. O relatório é substituído via status.tmp a cada ciclo; épocas e
    gráficos só aparecem nos logs dos workers. Não há heartbeat separado.
    """
    ctx = mp.get_context("spawn")
    root = Path(args.run_dir or f"queue_runs/{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=False)
    records = [dict(id=i, job=job, state="pending", attempts=0,
                    memory_mb=args.gpu_memory_mb, since=time.monotonic())
               for i, job in enumerate(jobs)]
    active = {}

    def report():
        """Publique todos os registros via substituição de arquivo temporário.

        Captura ``root`` e ``records`` do agendador. Escreve status.tmp com
        indentação de dois espaços e o renomeia para status.json no mesmo
        diretório, evitando expor uma escrita parcial aos leitores do destino.
        Retorna None e propaga erros de serialização ou de sistema de arquivos;
        não executa fsync nem coordena escritores externos.
        """
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(records, indent=2))
        temporary.replace(root / "status.json")

    print(f"Fila: {len(jobs)} execuções; relatório em {root}")
    try:
        while any(r["state"] in {"pending", "running"} for r in records):
            for ident, (process, gpu) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                rec = records[ident]
                status = root / f"{ident}.json"
                result = json.loads(status.read_text()) if status.exists() else {
                    "state": "failed", "error": f"Worker saiu com código {process.exitcode}"}
                rec.update(result)
                if rec["state"] == "oom" and rec["attempts"] <= args.oom_retries:
                    rec.update(state="pending", since=time.monotonic(), memory_mb=rec["memory_mb"] * 2)
                elif rec["state"] == "oom":
                    rec["state"] = "failed"
                print(f"Job {ident}: {rec['state']}")
                del active[ident]
            gpus = gpu_memory() if args.device == "cuda" else []
            for rec in records:
                if rec["state"] != "pending":
                    continue
                if time.monotonic() - rec["since"] >= args.wait_timeout:
                    rec.update(state="failed", error="Tempo limite esperando recursos")
                    continue
                if len(active) >= args.max_parallel:
                    continue
                gpu = None
                if args.device == "cuda":
                    # Conservative reservations also cover workers still initializing.
                    candidates = [(free - sum(records[i]["memory_mb"] for i, (_, g) in active.items() if g == uid), uid)
                                  for uid, free, total in gpus]
                    candidates.sort(reverse=True)
                    if not candidates or candidates[0][0] < rec["memory_mb"] + args.gpu_reserve_mb:
                        continue
                    gpu = candidates[0][1]
                ident = rec["id"]
                status = root / f"{ident}.json"
                if status.exists():
                    status.unlink()
                process = ctx.Process(target=worker, args=(rec["job"], gpu, str(status), str(root / f"{ident}.log")))
                process.start()
                active[ident] = (process, gpu)
                rec.update(state="running", attempts=rec["attempts"] + 1, gpu=gpu)
                print(f"Job {ident}: iniciado em {gpu or 'cpu'}")
            report()
            if any(r["state"] in {"pending", "running"} for r in records):
                time.sleep(min(args.poll_seconds, 30))
    except BaseException as exc:
        for process, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
        for rec in records:
            if rec["state"] in {"pending", "running"}:
                rec.update(state="interrupted", error=repr(exc))
        report()
        raise
    return [r for r in records if r["state"] != "completed"]
