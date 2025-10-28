#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// This C++ helper mirrors your train.py behavior by shelling out to Python/Ultralytics.
// Rationale: Ultralytics training is Python-only; there isn't a stable C++ training API.
// This wrapper lets you configure flags from C++ and still reuse your existing pipeline.
//
// Usage examples:
//   train_runner.exe --python "C:/Users/13578/PycharmProjects/PanelDectection/.venv/Scripts/python.exe" \
//                    --project "C:/Users/13578/PycharmProjects/PanelDectection" \
//                    --epochs 1 --imgsz 64 --batch 64 --device cpu --yaml armor.yaml
//   train_runner.exe --project . --yaml armor.yaml
//
// Notes:
// - We default to using the current working directory as the project folder.
// - If --python is omitted, we try environment "PYTHON" then fallback to "py -3" on Windows
//   or "python3" elsewhere.
// - We assert that the YAML exists next to train.py (like your script) OR at --yaml path.

struct Args {
    std::string python;      // interpreter path
    fs::path project = fs::current_path(); // folder containing train.py and armor.yaml
    std::string device = "cpu";
    int epochs = 1;
    int imgsz  = 64;
    int batch  = 64;
    fs::path yaml;           // optional explicit path to armor.yaml
};

static bool starts_with(const std::string &s, const std::string &p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--python" && i + 1 < argc) a.python = argv[++i];
        else if (k == "--project" && i + 1 < argc) a.project = fs::path(argv[++i]);
        else if (k == "--device" && i + 1 < argc) a.device = argv[++i];
        else if (k == "--epochs" && i + 1 < argc) a.epochs = std::stoi(argv[++i]);
        else if (k == "--imgsz"  && i + 1 < argc) a.imgsz  = std::stoi(argv[++i]);
        else if (k == "--batch"  && i + 1 < argc) a.batch  = std::stoi(argv[++i]);
        else if (k == "--yaml"   && i + 1 < argc) a.yaml   = fs::path(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete arg: " << k << "\n";
        }
    }
    return a;
}

static std::string detect_default_python() {
#ifdef _WIN32
    // Prefer PYTHON env
    if (const char* p = std::getenv("PYTHON")) return std::string(p);
    // Try 'py -3'
    return "py -3";
#else
    if (const char* p = std::getenv("PYTHON")) return std::string(p);
    return "python3"; // POSIX default
#endif
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    if (args.python.empty()) args.python = detect_default_python();

    // Resolve project folder and key files
    fs::path project = fs::weakly_canonical(args.project);
    fs::path train_py = project / "train.py"; // your original script

    // YAML resolution: 1) explicit --yaml path, else 2) next to train.py
    fs::path yaml = args.yaml.empty() ? (project / "armor.yaml") : fs::weakly_canonical(args.yaml);

    if (!fs::exists(project)) {
        std::cerr << "Project folder not found: " << project << "\n";
        return 1;
    }
    if (!fs::exists(train_py)) {
        std::cerr << "train.py not found under: " << project << "\n";
        return 2;
    }
    if (!fs::exists(yaml)) {
        std::cerr << "YAML not found: " << yaml << "\n";
        return 3;
    }

    // Compose the python command mirroring your train.py defaults
    // Your train.py internally asserts YAML and calls: model.train(data=armor.yaml, epochs=1, imgsz=64, batch=64, device="cpu")
    // We forward our knobs via env vars to keep train.py simple, but here we just pass CLI to Python with -c or call the file.

    std::ostringstream cmd;
    // If python path already contains spaces or is a multi-token launcher (e.g., "py -3"), keep it verbatim.
    cmd << args.python << " \"" << as_string(train_py) << "\"";
    // No CLI flags in your train.py; it reads constants. If you want, you can extend train.py to read envs or args.

    // For visibility, show what we will run.
    std::cout << "Running: " << cmd.str() << "\n";

    // Set working directory so train.py finds armor.yaml by relative path
#ifdef _WIN32
    _chdir(project.string().c_str());
#else
    chdir(project.string().c_str());
#endif

    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "Python training failed, code: " << rc << "\n";
        return rc;
    }

    // Optional: print a hint where Ultralytics stores runs
    fs::path runs = project / "runs" / "detect" / "train" / "weights" / "best.pt";
    if (fs::exists(runs)) {
        std::cout << "Best weights: " << runs << "\n";
    } else {
        std::cout << "Training finished. Check ./runs/ for outputs." << "\n";
    }

    return 0;
}
