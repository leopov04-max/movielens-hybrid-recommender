# Hệ thống gợi ý phim MovieLens

Đề tài cuối kỳ môn Học máy ứng dụng. Hệ thống phân loại từng lượt chấm của người dùng thành hai nhóm. Từ 4 sao trở lên là thích (`1`), dưới 4 sao là `0`. Mô hình dùng các đặc trưng về phim, hồ sơ người dùng và lịch sử đánh giá để xếp hạng Top-K phim.

## Chạy trên Windows

Yêu cầu Python **3.11**. Tại thư mục repo, chạy trong PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.prepare_data --tmdb
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Trên macOS hoặc Linux, dùng `.venv/bin/python` cho các lệnh sau bước tạo môi trường. Bỏ cờ `--tmdb` nếu chỉ cần huấn luyện và gợi ý. Nếu tải tự động gặp lỗi, xem [vị trí đặt dữ liệu](data/README.md). Sau khi app chạy, mở địa chỉ do Streamlit in ra, thường là `http://localhost:8501`.

Trong app, chọn một người dùng mẫu hoặc chấm vài phim để tạo hồ sơ mới. Hồ sơ chưa có lượt chấm nhận danh sách theo độ phổ biến. Tab **Đánh giá mô hình** hiển thị kết quả cho người dùng có lịch sử và người dùng mới.

Chạy `python -m unittest discover -s tests -v` trong môi trường ảo để kiểm tra quy tắc tạo đặc trưng theo thời gian.

## Thực nghiệm

- Chia các lượt chấm theo hai mốc thời gian chung. Giữ riêng khoảng 20% ID người dùng. Lượt chấm của các ID này không tham gia huấn luyện hoặc tạo lịch sử. Tập cold-start gồm các ID được giữ riêng xuất hiện trong giai đoạn test và các người dùng lần đầu xuất hiện trong giai đoạn đó.
- Với mỗi lượt train, chỉ dùng các lượt chấm trước thời điểm đó. Những lượt cùng giây dùng cùng một trạng thái lịch sử. Dùng ảnh chụp lịch sử cuối train cho validation và cuối validation cho test. Các bước điền thiếu, chuẩn hóa và mã hóa được fit trên train.
- Đặc trưng gồm 19 cờ thể loại, năm phát hành, điểm trung bình và số lượt chấm trong quá khứ của phim, tuổi, giới tính, nghề nghiệp, điểm trung bình và số lượt chấm của người dùng, cùng cosine similarity giữa thể loại phim và các phim họ từng thích.
- So sánh Logistic Regression L2, KNN và CART trên cùng tập chia. Chọn ngưỡng phân loại theo F1 trên validation, ưu tiên Precision khi bằng điểm. Chọn cấu hình và mô hình theo Precision@5 trên validation, rồi F1 và Precision khi bằng điểm. Báo cáo thêm baseline xếp theo số lượt chấm.
- Tính Precision@K trung bình theo người dùng, trên các phim có lượt chấm trong tập đánh giá. Chỉ tính người có ít nhất K ứng viên và báo số lượng đi kèm. Ứng dụng gợi ý trên các phim người dùng chưa chấm.

Xem [báo cáo thực nghiệm](reports/bao-cao.md) và [notebook khám phá](notebooks/01_kham_pha_du_lieu.ipynb). Khi huấn luyện, mô hình được lưu tại `artifacts/recommender.joblib` và kết quả chi tiết ở `reports/generated/`. Để mở notebook, cài JupyterLab trong cùng môi trường và khởi động từ thư mục repo.

## Bố cục mã nguồn

| Đường dẫn | Nội dung |
|---|---|
| `src/prepare_data.py` | Tải và đọc dữ liệu |
| `src/features.py` | Chia tập và tạo đặc trưng theo thời gian |
| `src/train.py`, `src/evaluate.py` | Huấn luyện, so sánh và xuất kết quả |
| `src/recommend.py`, `app.py` | Tạo Top-K, ghép metadata và hiển thị demo |

Dữ liệu chính là [MovieLens 100K của GroupLens](https://grouplens.org/datasets/movielens/100k/). Metadata bổ sung đến từ [TMDB 5000 trên Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Repo chỉ theo dõi mã nguồn và kết quả tổng hợp. Dữ liệu gốc và artifact huấn luyện được tạo cục bộ.
