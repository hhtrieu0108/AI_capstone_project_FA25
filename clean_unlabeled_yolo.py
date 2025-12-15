import argparse
from pathlib import Path

# Các đuôi ảnh phổ biến trong dataset YOLO
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_unlabeled(label_path: Path) -> bool:
    """
    Ảnh được coi là "không có label" nếu:
      - Không có file .txt tương ứng
      - Hoặc file .txt trống / chỉ có dòng trắng / comment
    """
    if not label_path.exists():
        return True

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # Nếu không đọc được (encoding lỗi) -> coi như không hợp lệ
        return True

    if not text:
        return True

    # Nếu tất cả các dòng đều trống hoặc bắt đầu bằng '#'
    lines = [ln.strip() for ln in text.splitlines()]
    valid_lines = [ln for ln in lines if ln and not ln.startswith("#")]

    return len(valid_lines) == 0


def process_split(root: Path, split: str, dry_run: bool = True):
    """
    Kiểm tra và (tuỳ chọn) xoá ảnh không có label cho một split (train/val/test)
    với cấu trúc:
      root/train/images
      root/train/labels
      root/val/images
      root/val/labels
      root/test/images
      root/test/labels
    """
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.exists():
        print(f"[{split}] ⚠️ Thư mục ảnh không tồn tại: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"[{split}] ⚠️ Thư mục nhãn không tồn tại: {labels_dir}")
        return

    print(f"\n===== Split: {split} =====")
    print(f"Ảnh:   {images_dir}")
    print(f"Nhãn:  {labels_dir}")

    image_paths = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    total_images = len(image_paths)
    unlabeled_images = []

    for img_path in image_paths:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if is_unlabeled(label_path):
            unlabeled_images.append((img_path, label_path))

    print(f"Tổng số ảnh: {total_images}")
    print(f"Ảnh không có label (hoặc label trống): {len(unlabeled_images)}")

    if not unlabeled_images:
        print(f"[{split}] ✅ Không có ảnh nào cần xoá.")
        return

    # Liệt kê vài ảnh đầu tiên để kiểm tra
    print("\nMột số ảnh không có label:")
    for img_path, label_path in unlabeled_images[:10]:
        print(f"  - Ảnh : {img_path}")
        if label_path.exists():
            print(f"    Label (trống/không hợp lệ): {label_path}")
        else:
            print(f"    Label: (KHÔNG TỒN TẠI) {label_path}")

    if dry_run:
        print(f"\n[{split}] 🔍 Đang ở chế độ dry-run, KHÔNG xoá file.")
        print(f"    Thêm --apply nếu muốn xoá thật.")
        return

    # Thực sự xoá file nếu không phải dry_run
    print(f"\n[{split}] ❌ Xoá ảnh không có label ...")
    deleted_imgs = 0
    deleted_lbls = 0

    for img_path, label_path in unlabeled_images:
        try:
            if img_path.exists():
                img_path.unlink()
                deleted_imgs += 1
        except Exception as e:
            print(f"  ⚠️ Lỗi xoá ảnh {img_path}: {e}")

        # Nếu có label file (dù trống) thì xoá luôn cho sạch
        if label_path.exists():
            try:
                label_path.unlink()
                deleted_lbls += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi xoá nhãn {label_path}: {e}")

    print(f"[{split}] ✅ Đã xoá {deleted_imgs} ảnh và {deleted_lbls} file nhãn trống/không hợp lệ.")


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và xoá ảnh không có label trong dataset YOLO (format: root/split/images, root/split/labels)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Đường dẫn tới thư mục gốc dataset (chứa train/, val/, test/...).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Danh sách các split cần xử lý, ví dụ: --splits train val test",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Thực sự xoá file. Mặc định không xoá (dry-run).",
    )

    args = parser.parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"❌ Thư mục root không tồn tại: {root}")
        return

    dry_run = not args.apply
    if dry_run:
        print("🔍 Đang chạy ở chế độ DRY-RUN (KHÔNG xoá file).")
        print("    Khi kiểm tra xong, thêm cờ --apply để xoá thật.\n")

    for split in args.splits:
        process_split(root, split, dry_run=dry_run)


if __name__ == "__main__":
    main()
