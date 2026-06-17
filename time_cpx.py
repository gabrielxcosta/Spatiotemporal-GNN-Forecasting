from utils.computational_time import ComputationalTimeAnalyzer

if __name__ == "__main__":
    roots = [
        "results_chickenpox",
        "results_wikimaths",
        "results_englandcovid",
        "results_montevideobus",
        "results_chickenpox_long",
        "results_wikimaths_long",
        "results_englandcovid_long",
        "results_montevideobus_long",
    ]

    analyzer = ComputationalTimeAnalyzer(roots)

    analyzer.load()

    analyzer.export_tables(
        output_dir="computational_time",
    )

    analyzer.print_topk_fastest_models(
        k=5,
    )

    analyzer.print_model_regime_summary()

    analyzer.print_model_overall_summary()

    analyzer.print_family_summaries()

    analyzer.plot_by_model_regime(
        output_dir="computational_time/plots",
    )

    analyzer.plot_by_model_overall(
        output_dir="computational_time/plots",
    )
