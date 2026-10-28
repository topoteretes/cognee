"""Resolve internal imports only against source files present in the snapshot."""

import posixpath
from collections import defaultdict

_SCRIPT_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_TYPESCRIPT_EXTENSIONS = {".ts", ".tsx", ".mts", ".cts"}


class ImportIndex:
    """Modules, source files and Go module paths of a snapshot, indexed once for all lookups."""

    def __init__(self, facts, fallback_repo):
        self.modules = {}
        self.files = defaultdict(set)
        self.lookup = defaultdict(set)
        self.module_path_repos = defaultdict(set)
        module_paths = defaultdict(set)
        for fact in facts:
            repo = fact.get("repo")
            if not isinstance(repo, str) or not repo:
                repo = fallback_repo
            if fact.get("kind") == "module" and isinstance(fact.get("name"), str):
                self.modules[(repo, fact["name"])] = fact
                props = fact.get("props") or {}
                module_path = props.get("modulePath") if isinstance(props, dict) else None
                if isinstance(module_path, str) and module_path:
                    module_paths[repo].add(module_path)
                    self.module_path_repos[module_path].add(repo)
            file = fact.get("file")
            if fact.get("kind") in ("symbol", "file_ref") and isinstance(file, str) and file:
                self.files[repo].add(file)

        self.module_paths = dict(module_paths)

        for repo, files in self.files.items():
            for file in files:
                self.lookup[(repo, "exact", file)].add(file)
                stem, suffix = posixpath.splitext(file)
                aliases = {stem}
                if posixpath.basename(stem) in {"index", "__init__", "mod"}:
                    aliases.add(posixpath.dirname(stem))
                for alias in aliases:
                    self.lookup[(repo, suffix, alias)].add(file)

    def module_path_prefixes(self, name, separators, repo=None):
        """The module paths that start ``name`` and end before a separator, longest first.

        One lookup per separator in ``name``, whatever the number of module paths.
        """
        known = self.module_path_repos if repo is None else self.module_paths.get(repo, ())
        for index in range(len(name) - 1, 0, -1):
            if name[index] in separators and name[:index] in known:
                yield name[:index]

    def module_for_file(self, repo, file):
        directory = file if (repo, file) in self.modules else posixpath.dirname(file) or "."
        while True:
            if (repo, directory) in self.modules:
                return self.modules[(repo, directory)]
            if directory in ("", "."):
                return None
            directory = posixpath.dirname(directory) or "."

    def resolve(self, fact, target, repo):
        props = fact.get("props") or {}
        if not isinstance(props, dict) or props.get("source") != "internal":
            return None
        language = props.get("language")
        source = fact.get("file")
        if not isinstance(source, str):
            return None
        if language == "go":
            for prefix in self.module_path_prefixes(target, "/", repo):
                target = target[len(prefix) + 1 :]
                break
        elif language in {"java", "rust"}:
            target = target.replace("::", "/") if language == "rust" else target.replace(".", "/")
            if language == "rust" and target.startswith("crate/"):
                target = "src/" + target[len("crate/") :]
        if target.startswith(("./", "../")):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
        elif language == "python" and "/" not in target:
            # Enola normally emits a slash-qualified target; older snapshots
            # can contain Python's dotted import notation.
            if target.startswith("."):
                depth = len(target) - len(target.lstrip("."))
                base = posixpath.dirname(source)
                for _ in range(depth - 1):
                    base = posixpath.dirname(base) or ".."
                target = posixpath.normpath(posixpath.join(base, target[depth:].replace(".", "/")))
            else:
                target = target.replace(".", "/")
        if target.startswith(("../", "/")):
            return None
        if (repo, target) in self.modules:
            return self.modules[(repo, target)]
        extensions = {
            "python": {".py", ".pyi"},
            "typescript": {".ts", ".tsx", ".js", ".jsx", ".mts", ".cts"},
            "javascript": {".js", ".jsx", ".mjs", ".cjs"},
            "go": {".go"},
            "java": {".java"},
            "rust": {".rs"},
        }.get(language, set())
        files = set(self.lookup.get((repo, "exact", target), ()))
        for extension in extensions:
            files.update(self.lookup.get((repo, extension, target), ()))
        stem, suffix = posixpath.splitext(target)
        if not files and suffix in _SCRIPT_EXTENSIONS and language in {"typescript", "javascript"}:
            # An ESM import of ``./a.js`` names the compiled file of ``a.ts``.
            for extension in _TYPESCRIPT_EXTENSIONS:
                files.update(self.lookup.get((repo, extension, stem), ()))
        if len(files) != 1:
            return None
        return self.module_for_file(repo, next(iter(files)))
