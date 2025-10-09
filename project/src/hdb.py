from pathlib import Path

from openpyxl.styles.builtins import output

from project.src.data_processing import read_files, train_test_process
from project.src.visualization import plot_figures

from project.src import add_features, model_train


def main():
    data_dir = Path("../data")
    # train_df, test_df, auxiliary = read_files(data_dir)
    # train_df = train_test_process(train_df)
    # test_df = train_test_process(test_df)
    # plot_figures(train_df)
    # train_df.to_csv(data_dir / 'train_processed.csv', index=False)
    # test_df.to_csv(data_dir / '/test_processed.csv', index=False)


    output_dir = Path("../data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # add_features.main(
    #     train_path=data_dir / "train_processed.csv",
    #     test_path=data_dir / "test_processed.csv",
    #     aux_dir=data_dir / "aux",
    #     out_train=output_dir / "train_with_all_features.csv",
    #     out_test=output_dir / "test_with_all_features.csv",
    #     radius_km=1
    # )

    model_train.main(output_dir)

if __name__ == "__main__":
    main()
