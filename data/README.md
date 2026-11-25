# Dữ liệu cục bộ

Chạy `python -m src.prepare_data --tmdb` từ thư mục repo. Nếu tải thủ công, giải nén đúng cấu trúc dưới đây. TMDB là tùy chọn và chỉ dùng để hiển thị thông tin phim.

```text
data/raw/
├── ml-100k/
│   ├── u.data
│   ├── u.item
│   └── u.user
└── tmdb/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

- [MovieLens 100K và điều kiện sử dụng của GroupLens](https://grouplens.org/datasets/movielens/100k/)
- [TMDB 5000 trên Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
