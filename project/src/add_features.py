# add_feature.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

R_KM = 6371  # 地球半径（km）

# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def add_location_features(df: pd.DataFrame,
                          ref_df: pd.DataFrame,
                          prefix: str,
                          radius_km: float = 1.0) -> pd.DataFrame:
    """
    通用空间特征生成器：最近距离 & radius_km 内个数
    ref_df 必须含 LATITUDE / LONGITUDE
    """
    ref_df = ref_df.dropna(subset=["LATITUDE", "LONGITUDE"])
    if ref_df.empty:
        df[f"nearest_{prefix}_distance"] = np.nan
        df[f"{prefix}_within_{radius_km:.0f}km"] = np.nan
        return df

    tree = BallTree(np.radians(ref_df[["LATITUDE", "LONGITUDE"]]),
                    metric="haversine")
    radius_rad = radius_km / R_KM

    df = df.copy()
    df[f"nearest_{prefix}_distance"] = np.nan
    df[f"{prefix}_within_{radius_km:.0f}km"] = np.nan

    mask = df[["LATITUDE", "LONGITUDE"]].notna().all(axis=1)
    if not mask.any():
        return df

    hdb_rad = np.radians(df.loc[mask, ["LATITUDE", "LONGITUDE"]])
    dist_rad, _ = tree.query(hdb_rad, k=1)
    count = tree.query_radius(hdb_rad, r=radius_rad, count_only=True)

    df.loc[mask, f"nearest_{prefix}_distance"] = dist_rad[:, 0] * R_KM
    df.loc[mask, f"{prefix}_within_{radius_km:.0f}km"] = count.astype(float)
    return df


def add_floor_position(df: pd.DataFrame) -> pd.DataFrame:
    """FLOOR_POSITION = FLOOR / MAX_FLOOR，范围 [0,1]"""
    df = df.copy()
    df["FLOOR_POSITION"] = np.nan
    mask = df[["FLOOR", "MAX_FLOOR"]].notna().all(axis=1) & (df["MAX_FLOOR"] > 0)
    df.loc[mask, "FLOOR_POSITION"] = (
        df.loc[mask, "FLOOR"] / df.loc[mask, "MAX_FLOOR"]
    ).clip(0, 1)
    return df


# ------------------------------------------------------------------
# 主封装函数
# ------------------------------------------------------------------
def main(train_path: Path ,
         test_path: Path ,
         aux_dir: Path ,
         out_train: Path ,
         out_test: Path ,
         radius_km: float = 1.0):
    """
    读取原始文件 → 加空间特征 + FLOOR_POSITION → 保存结果
    所有路径均可外部传入
    """
    aux = Path(aux_dir)

    # 1. 读取
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    landmarks = {
        "hawker": pd.read_csv(aux / "sg-gov-hawkers.csv"),
        "mrt": pd.read_csv(aux / "sg-mrt-stations.csv"),
        "primary_school": pd.read_csv(aux / "sg-primary-schools.csv"),
        "secondary_school": pd.read_csv(aux / "sg-secondary-schools.csv"),
        "mall": pd.read_csv(aux / "sg-shopping-malls.csv"),
        # HDB 坐标
        "block": pd.read_csv(aux / "sg-hdb-block-details.csv")[
            ["TOWN", "BLOCK", "LATITUDE", "ADDRESS", "LONGITUDE", "MAX_FLOOR"]
        ],
    }

    # 2. 合并坐标 & MAX_FLOOR —— 只 pop 一次
    block_coords = landmarks.pop("block")
    block_coords.rename(columns={'ADDRESS': 'STREET'}, inplace=True)
    train = train.merge(block_coords, on=['TOWN', 'BLOCK', 'STREET'], how='left')
    test = test.merge(block_coords, on=['TOWN', 'BLOCK', 'STREET'], how='left')

    # 3. 加空间特征
    for prefix, ref in landmarks.items():
        print(f"Processing {prefix} ...")
        train = add_location_features(train, ref, prefix, radius_km=radius_km)
        test = add_location_features(test, ref, prefix, radius_km=radius_km)

    # 4. 加楼层相对位置
    train = add_floor_position(train)
    test = add_floor_position(test)

    # 5. 保存
    train.to_csv(out_train, index=False, float_format="%.6f")
    test.to_csv(out_test, index=False, float_format="%.6f")
    print(f"✅ Saved → {out_train}  &  {out_test}")


# ------------------------------------------------------------------
# 只有当脚本被直接运行时才会触发 main()
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
