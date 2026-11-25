import json

import joblib
import pandas as pd
import streamlit as st

from src.prepare_data import ARTIFACTS, GENRES, MODEL_NAME, REPORTS, TMDB, TMDB_FILES
from src.recommend import available_movies, load_metadata, recommend

st.set_page_config(page_title="Gợi ý phim MovieLens", page_icon="🎬", layout="wide")


@st.cache_resource
def load_bundle(path, updated_at):
    return joblib.load(path)


@st.cache_data
def cached_metadata(movies, file_versions):
    return load_metadata(movies)


def show_recommendations(bundle, movies, metadata):
    controls, results = st.columns([1, 2], gap="large")
    with controls:
        source = st.radio("Hồ sơ", ["Người dùng mẫu", "Sở thích của tôi"], horizontal=True)
        top_k = st.slider("Số phim gợi ý", min_value=5, max_value=20, value=10)
        user_id = None
        ratings = None
        if source == "Người dùng mẫu":
            user_id = st.selectbox("Người dùng MovieLens", sorted(bundle["history"].users))
            seen = len(bundle["history"].users[user_id].rated)
            st.caption(f"Hồ sơ có {seen} lượt chấm trước mốc đánh giá.")
        else:
            options = movies.sort_values("title")["title"].to_dict()
            selected = st.multiselect("Phim bạn đã xem", options, format_func=options.get, max_selections=10)
            ratings = {
                movie_id: st.slider(options[movie_id], 1, 5, 4, key=f"rating_{movie_id}")
                for movie_id in selected
            }
            if not ratings:
                st.info("Chọn và chấm phim để cá nhân hóa. Danh sách hiện tại dựa trên độ phổ biến.")
    with results:
        items = recommend(bundle, user_id=user_id, ratings=ratings, top_k=top_k)
        st.subheader("Phim dành cho bạn")
        if items.empty:
            st.info("Hồ sơ này đã chấm toàn bộ phim trong danh mục.")
            return
        if "score" in items:
            st.caption(f"Xếp theo xác suất thích ước lượng · {bundle['model_name']}")
        for rank, (movie_id, movie) in enumerate(items.iterrows(), 1):
            with st.container(border=True):
                title, score = st.columns([4, 1])
                title.markdown(f"**{rank}. {movie['title']}**")
                if "score" in movie:
                    score.write(f"{movie['score']:.0%}")
                else:
                    score.caption(f"{movie['rating_count']} lượt chấm")
                genres = [genre for genre in GENRES[1:] if movie[genre]]
                if genres:
                    st.caption(" · ".join(genres))
                details = metadata.get(movie_id)
                if details:
                    with st.expander("Thông tin phim"):
                        if details["overview"]:
                            st.write(details["overview"])
                        if details["cast"]:
                            st.caption("Diễn viên · " + ", ".join(details["cast"]))


def show_results(coverage):
    metrics_path = REPORTS / "metrics.json"
    if not metrics_path.is_file():
        st.info("Chạy `python -m src.train` để tạo kết quả đánh giá.")
        return
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    cohort = st.radio("Nhóm kiểm thử", ["Có lịch sử", "Người dùng mới"], horizontal=True)
    group = "warm_test" if cohort == "Có lịch sử" else "cold_test"
    labels = {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "precision_at_5": "Precision@5", "precision_at_10": "Precision@10",
    }
    rows = {
        model: {label: metrics[group][key] for key, label in labels.items()}
        for model, metrics in report["models"].items()
    }
    baseline = report["popularity_baseline"][group]
    rows["Độ phổ biến"] = {labels[key]: baseline[key] for key in ("precision_at_5", "precision_at_10")}
    st.dataframe(pd.DataFrame.from_dict(rows, orient="index").round(3), width="stretch")
    st.caption(f"Precision@5 tính trên {baseline['users_at_5']} người dùng có ít nhất 5 phim "
               f"đã chấm trong tập kiểm thử. Precision@10 dùng {baseline['users_at_10']} người.")
    selected = report["selected_model"]
    values = report["models"][selected][group]["confusion_matrix"]
    st.subheader(f"Ma trận nhầm lẫn · {selected}")
    matrix = pd.DataFrame(values, index=["Thực tế 0", "Thực tế 1"], columns=["Dự đoán 0", "Dự đoán 1"])
    st.dataframe(matrix, width="stretch")
    if coverage is not None:
        st.caption(f"TMDB có thông tin cho {coverage['matched']}/{coverage['total']} phim MovieLens.")


def main():
    st.title("Gợi ý phim MovieLens")
    path = ARTIFACTS / MODEL_NAME
    if not path.is_file():
        st.info("Tải MovieLens và huấn luyện mô hình để bắt đầu.")
        st.code("python -m src.prepare_data --tmdb\npython -m src.train", language="bash")
        return
    bundle = load_bundle(path, path.stat().st_mtime_ns)
    versions = tuple((TMDB / file).stat().st_mtime_ns if (TMDB / file).is_file() else None
                     for file in TMDB_FILES)
    metadata, coverage = cached_metadata(bundle["movies"], versions)
    recommendations, evaluation = st.tabs(["Gợi ý phim", "Đánh giá mô hình"])
    with recommendations:
        show_recommendations(bundle, available_movies(bundle), metadata)
    with evaluation:
        show_results(coverage)


if __name__ == "__main__":
    main()
