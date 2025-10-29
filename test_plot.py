from networks.TemporalNetworkAnalyzer import TemporalNetworkAnalyzer

if __name__ == "__main__":
    analyzer = TemporalNetworkAnalyzer(
        dataset_name="ChickenpoxHungary",
        base_dir="networks",
        geojson_path="networks/Layouts/hu.json"  # ajuste conforme sua estrutura
    )
    combined = analyzer.analyze_temporal_metrics()
    analyzer.plot_metric_evolution(combined)
    analyzer.plot_degree_heatmap(combined)
    analyzer.visualize_static_network(snapshot_idx=0)