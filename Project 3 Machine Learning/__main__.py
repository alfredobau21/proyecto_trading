from ML_Test import Operation, TradingStrategy

if __name__ == "__main__":
    strategy = TradingStrategy(train_path="data/aapl_project_train.csv", test_path="data/aapl_project_test.csv")
    strategy.train_model()
    strategy.plot_results()