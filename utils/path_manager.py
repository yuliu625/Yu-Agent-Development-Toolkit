from pathlib import Path


class PathManager:
    def __init__(self, path: str):
        self.path = Path(path)

    def recursive_find_files_at_depth(
        self,
        root_dir: str,
        depth: int,
        file_names: list = None,
    ) -> list:
        root_dir = Path(root_dir)
        result = []
        for p in root_dir.rglob('*'):
            if p.is_file() and len(p.relative_to(root_dir).parts) == depth+1 and p.name in file_names:
                result.append(p)
        return result


if __name__ == '__main__':
    pass
