from pathlib import Path
import shutil


def safe_mkdir(directory: str | Path) -> Path:
    directory_path = Path(directory)
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Failed to create directory {directory_path}: {e}"
        ) from e
    return directory_path


def get_file_list(
    directory: str | Path, valid_extensions: list[str]
) -> list[str]:
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(
            f"Directory not found or is not a directory: {directory}"
        )

    processed_extensions = []
    for ext in valid_extensions:
        ext_lower = ext.lower()
        if not ext_lower.startswith("."):
            processed_extensions.append(f".{ext_lower}")
        else:
            processed_extensions.append(ext_lower)

    file_list = []
    for ext_pattern in processed_extensions:
        file_list.extend(
            str(f.resolve()) for f in directory_path.rglob(f"*{ext_pattern}")
        )
    return file_list


def copy_with_structure(
    src_file: str | Path, src_root: str | Path, dst_root: str | Path
) -> str:
    src_path = Path(src_file).resolve()
    src_root_path = Path(src_root).resolve()
    dst_root_path = Path(dst_root).resolve()

    if not src_path.is_file():
        raise FileNotFoundError(
            f"Source file not found or is not a file: {src_file}"
        )
    if not src_root_path.is_dir():
        raise NotADirectoryError(
            f"Source root is not a directory: {src_root}"
        )

    try:
        rel_path = src_path.parent.relative_to(src_root_path)
    except ValueError as e:
        raise ValueError(
            f"Source file's parent '{src_path.parent}' is not a subdirectory "
            f"of src_root '{src_root_path}'."
        ) from e

    dst_dir = dst_root_path / rel_path
    safe_mkdir(dst_dir)

    dst_file_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_file_path)

    return str(dst_file_path)
