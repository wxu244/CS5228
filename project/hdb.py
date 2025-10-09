from project.data_processing import read_files, train_test_process
from project.visualization import plot_figures


def main():
    train_df, test_df, auxiliary = read_files()
    train_df = train_test_process(train_df)
    test_df = train_test_process(test_df)
    plot_figures(train_df)
    train_df.to_csv('train_after_process.csv', index=False)
    test_df.to_csv('test_after_process.csv', index=False)
    return

if __name__ == "__main__":
    main()
