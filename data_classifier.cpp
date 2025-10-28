#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static std::string to_lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string stem_of(const fs::path &p) {
    return p.stem().string();
}

static std::string as_posix(const fs::path &p) {
    std::string s = p.string();
    for (char &c : s) if (c == '\\') c = '/';
    return s;
}

int main(int argc, char **argv) {
    // Defaults mirror your Python script
    fs::path root = argc > 1 ? fs::path(argv[1]) : fs::path("C:/Users/13578/Downloads/armor_dataset_v4");
    double split = argc > 2 ? std::stod(argv[2]) : 0.9;  // train ratio

    fs::path img_dir = root / "images";
    fs::path lbl_dir = root / "labels";
    fs::path lists_dir = root / "lists";

    std::set<std::string> IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};

    if (!fs::exists(img_dir) || !fs::exists(lbl_dir)) {
        std::cerr << "ERROR: images/ or labels/ folder not found under: " << root << "\n";
        return 1;
    }
    fs::create_directories(lists_dir);

    // Gather image files
    std::vector<fs::path> image_files;
    for (auto const &e : fs::recursive_directory_iterator(img_dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = to_lower(e.path().extension().string());
        if (IMG_EXTS.count(ext)) image_files.push_back(e.path());
    }

    // Gather label files (.txt)
    std::vector<fs::path> label_files;
    for (auto const &e : fs::recursive_directory_iterator(lbl_dir)) {
        if (!e.is_regular_file()) continue;
        if (to_lower(e.path().extension().string()) == ".txt") label_files.push_back(e.path());
    }

    // Map by stem
    std::unordered_map<std::string, fs::path> img_by_stem, lbl_by_stem;
    for (auto const &p : image_files) img_by_stem[stem_of(p)] = p;
    for (auto const &p : label_files) lbl_by_stem[stem_of(p)] = p;

    // Intersect stems
    std::vector<std::string> common;
    common.reserve(std::min(img_by_stem.size(), lbl_by_stem.size()));
    for (auto const &kv : img_by_stem) if (lbl_by_stem.count(kv.first)) common.push_back(kv.first);
    std::sort(common.begin(), common.end());

    if (common.empty()) {
        std::cerr << "ERROR: No matched image/label pairs found." << std::endl;
        return 2;
    }

    // Build pairs
    std::vector<fs::path> imgs;
    imgs.reserve(common.size());
    for (auto const &s : common) imgs.push_back(img_by_stem[s]);

    // Deterministic shuffle with seed 0 (like Python random.seed(0))
    std::mt19937 rng(0);
    std::shuffle(imgs.begin(), imgs.end(), rng);

    // Train/val split
    size_t cut = static_cast<size_t>(imgs.size() * split);
    std::vector<fs::path> train_imgs(imgs.begin(), imgs.begin() + cut);
    std::vector<fs::path> val_imgs(imgs.begin() + cut, imgs.end());

    // Write lists (paths to images). test.txt == val.txt as in your script
    auto write_list = [&](const fs::path &out_path, const std::vector<fs::path> &paths) {
        std::ofstream ofs(out_path);
        if (!ofs) {
            std::cerr << "ERROR: Cannot write file: " << out_path << "\n";
            std::exit(3);
        }
        for (auto const &p : paths) ofs << as_posix(p) << '\n';
    };

    write_list(lists_dir / "train.txt", train_imgs);
    write_list(lists_dir / "val.txt",   val_imgs);
    write_list(lists_dir / "test.txt",  val_imgs);

    std::cout << "Wrote:\n  " << (lists_dir / "train.txt")
              << "\n  " << (lists_dir / "val.txt")
              << "\n  " << (lists_dir / "test.txt") << "\n";

    return 0;
}

