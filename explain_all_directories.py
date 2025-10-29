import os

IGNORED_DIRS = {
    "/media/work/gabrielcosta/pyg_env",
    "/media/work/gabrielcosta/myenv",
    "/media/work/gabrielcosta/.cache",
    "/media/work/gabrielcosta/results"
}

def classify_folder(path):
    """Classifica uma pasta de acordo com o conteúdo detectado."""
    try:
        files = os.listdir(path)
    except PermissionError:
        return "🚫 Sem permissão para acessar"
    
    if any(f.endswith(('.csv', '.npz', '.json', '.pt', '.pkl')) for f in files):
        return "🧩 Dados (datasets espaciais-temporais ou grafos)"
    if any(f.endswith(('.py', '.ipynb')) for f in files):
        return "⚙️ Código (modelos, treino, ou experimentos)"
    if any(f.endswith(('.txt', '.pdf', '.md', '.docx')) for f in files):
        return "📚 Documentação / Artigos"
    if any(f.endswith(('.png', '.jpg', '.svg')) for f in files):
        return "📊 Resultados / Visualizações"
    return "📦 Pasta auxiliar ou vazia"

def describe_project_structure(base_dir="."):
    base_dir = os.path.abspath(base_dir)
    print(f"\n📁 Estrutura do projeto em: {base_dir}\n")

    for root, dirs, files in os.walk(base_dir):
        if any(root.startswith(ignored) for ignored in IGNORED_DIRS):
            dirs[:] = []
            continue

        level = root.replace(base_dir, "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root) or "pasta raiz"

        descricao = classify_folder(root)
        print(f"{indent}📦 {folder_name} → {descricao}")

        subindent = "  " * (level + 1)
        for f in files[:5]:  # mostra até 5 arquivos
            print(f"{subindent}📄 {f}")
        if files:
            print(f"{subindent}({len(files)} arquivos no total)\n")
        else:
            print(f"{subindent}(vazio)\n")

    print("\n🧠 Fluxo do projeto (seguindo o paper 'Exploring Neural Network Architectures for Time Series Prediction on Graphs'):")
    print("1️⃣ **Dados** → grafos e séries temporais (entrada).")
    print("2️⃣ **Modelos** → módulos espaciais e temporais (GNNs, Transformers, RNNs).")
    print("3️⃣ **Treino** → scripts de treino/validação e ablações.")
    print("4️⃣ **Resultados** → métricas (MAE, RMSE, MAPE), testes estatísticos e visualizações.")
    print("5️⃣ **Documentação** → relatórios e o paper principal.\n")

if __name__ == "__main__":
    describe_project_structure()
