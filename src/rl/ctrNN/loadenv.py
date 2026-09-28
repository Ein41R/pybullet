from pathlib import Path
import os  


class LoadEnv():
    def __init__(self) -> None:
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self._load_env_file()

    def _load_env_file(self) -> None:
        env_path = self.PROJECT_ROOT / ".env.development"
        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'") 
            if key and key not in os.environ:
                os.environ[key] = value

    # funcion resolves the path
    def _resolve_path(self, env_name: str, default_path: Path) -> Path:
        raw_value = os.getenv(env_name)
        if not raw_value:
            return default_path

        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (self.PROJECT_ROOT / candidate).resolve()
        return candidate
