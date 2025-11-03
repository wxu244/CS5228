from pathlib import Path

from catboost import CatBoostRegressor
from openpyxl.styles.builtins import output



import add_features, model_train
from project.src import catboost_model_all
from project.src.data_processing import *
from project.src.visualization import *



def main():
    data_dir = Path("../data")
    train_df, test_df, auxiliary = read_files(data_dir)
    train_df, test_df = train_test_process(train_df, test_df)
    # plot_figures(train_df)
    train_df.to_csv(data_dir / 'test' / 'train_processed_2.csv', index=False)
    test_df.to_csv(data_dir / 'test' / 'test_processed_2.csv', index=False)


    output_dir = Path("../data/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    add_features.main(
        train_path=data_dir / 'test' /  "train_processed_2.csv",
        test_path=data_dir / 'test' / "test_processed_2.csv",
        aux_dir=data_dir / "auxiliary-data",
        out_train=output_dir / "train_with_all_features.csv",
        out_test=output_dir / "test_with_all_features.csv",
        radius_km=1
    )

    # catboost_model_all.main(output_dir)
    model_train.main(output_dir)

if __name__ == "__main__":
    main()

    # output_dir = Path("../data/test")
    # model_train.main(output_dir)
