from utils.results import ResultsAnalyzer
import pandas as pd

if __name__ == "__main__":

    roots = [
        "results_chickenpox",
        "results_wikimaths",
        "results_englandcovid",
        "results_montevideobus",
    ]

    analyzer = ResultsAnalyzer(roots)
    analyzer.load()
    analyzer.plot_cd_subplots()