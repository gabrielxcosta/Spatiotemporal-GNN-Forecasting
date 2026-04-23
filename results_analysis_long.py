from utils.results_long import ResultsAnalyzer
import pandas as pd

if __name__ == "__main__":

    roots = [
        "results_chickenpox_long",
        "results_wikimaths_long",
        "results_englandcovid_long",
        "results_montevideobus_long",
    ]

    analyzer = ResultsAnalyzer(roots)
    analyzer.load()
    analyzer.plot_cd_subplots()