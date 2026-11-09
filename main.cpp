#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Structure pour stocker les métriques d'évaluation
struct Metrics {
    int contours_detectes = 0;
    int contours_reference = 0;
    int contours_corrects = 0;
    int faux_positifs = 0;
    int faux_negatifs = 0;
    
    double P = 0.0;   // Performance
    double TFP = 0.0; // Taux de Faux Positifs
    double TFN = 0.0; // Taux de Faux Négatifs
};

// Application des détecteurs de contours
cv::Mat applyDetector(const cv::Mat& src_gray, int detector_type, int p1, int p2, int p3) {
    cv::Mat edges;
    
    if (detector_type == 0) { // SOBEL
        int ksize = (p1 % 2 == 0) ? p1 + 1 : p1; // Le noyau doit être impair (1, 3, 5, 7)
        if (ksize < 1) ksize = 1;
        if (ksize > 7) ksize = 7;
        
        cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
        cv::Sobel(src_gray, grad_x, CV_16S, 1, 0, ksize);
        cv::Sobel(src_gray, grad_y, CV_16S, 0, 1, ksize);
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        
        // Seuil binaire pour extraire les contours
        cv::threshold(grad, edges, p2, 255, cv::THRESH_BINARY);
    } 
    else if (detector_type == 1) { // LAPLACE
        int ksize = (p1 % 2 == 0) ? p1 + 1 : p1;
        if (ksize < 1) ksize = 1;
        if (ksize > 7) ksize = 7;
        
        cv::Mat dst, abs_dst;
        cv::Laplacian(src_gray, dst, CV_16S, ksize);
        cv::convertScaleAbs(dst, abs_dst);
        
        cv::threshold(abs_dst, edges, p2, 255, cv::THRESH_BINARY);
    } 
    else if (detector_type == 2) { // CANNY
        int lowThreshold = p1;
        int highThreshold = std::max(p2, lowThreshold + 1);
        int apertureSize = (p3 % 2 == 0) ? p3 + 1 : p3;
        if (apertureSize < 3) apertureSize = 3;
        if (apertureSize > 7) apertureSize = 7;
        
        cv::Canny(src_gray, edges, lowThreshold, highThreshold, apertureSize);
    }
    
    return edges;
}

// Calcul des métriques avec une correspondance 1-a-1 dans un voisinage 3x3.
Metrics computeMetrics(const cv::Mat& detected_edges, const cv::Mat& ref_edges) {
    Metrics m;
    
    // S'assurer que les deux images sont binarisées (0 ou 255)
    cv::Mat det, ref;
    cv::threshold(detected_edges, det, 127, 255, cv::THRESH_BINARY);
    // Les références fournies contiennent des contours noirs sur fond blanc.
    cv::threshold(ref_edges, ref, 127, 255, cv::THRESH_BINARY_INV);
    
    m.contours_detectes = cv::countNonZero(det);
    m.contours_reference = cv::countNonZero(ref);

    // Chaque pixel détecté ne peut correspondre qu'à un seul pixel de référence.
    cv::Mat matched_reference = cv::Mat::zeros(ref.size(), CV_8U);
    for (int y = 0; y < det.rows; ++y) {
        for (int x = 0; x < det.cols; ++x) {
            if (det.at<uchar>(y, x) == 0) continue;

            bool matched = false;
            for (int dy = -1; dy <= 1 && !matched; ++dy) {
                for (int dx = -1; dx <= 1 && !matched; ++dx) {
                    int ref_y = y + dy;
                    int ref_x = x + dx;
                    if (ref_y >= 0 && ref_y < ref.rows && ref_x >= 0 && ref_x < ref.cols &&
                        ref.at<uchar>(ref_y, ref_x) != 0 &&
                        matched_reference.at<uchar>(ref_y, ref_x) == 0) {
                        matched_reference.at<uchar>(ref_y, ref_x) = 255;
                        matched = true;
                    }
                }
            }
            if (matched) ++m.contours_corrects;
        }
    }

    m.faux_positifs = m.contours_detectes - m.contours_corrects;
    m.faux_negatifs = m.contours_reference - m.contours_corrects;
    
    // Calcul des ratios
    double denom = static_cast<double>(m.contours_corrects + m.faux_positifs + m.faux_negatifs);
    if (denom > 0) {
        m.P = m.contours_corrects / denom;
        m.TFP = m.faux_positifs / denom;
        m.TFN = m.faux_negatifs / denom;
    }
    
    return m;
}

void runBatch(int detector_type, int param1, int param2, int param3,
              const std::string& img_dir, const std::string& gt_dir,
              const std::string& csv_path) {
    std::string detector_name = detector_type == 0 ? "Sobel" : (detector_type == 1 ? "Laplace" : "Canny");
    std::ofstream csv(csv_path);
    csv << "detecteur,image,contours_detectes,contours_reference,contours_corrects,faux_positifs,faux_negatifs,P,TFP,TFN\n";

    double sum_P = 0, sum_TFP = 0, sum_TFN = 0;
    int count = 0;
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        std::string stem = entry.path().stem().string();
        fs::path gt_path = fs::path(gt_dir) / (stem + "_gt_binary.jpg");
        cv::Mat gt = cv::imread(gt_path.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty() || gt.empty()) continue;

        Metrics m = computeMetrics(applyDetector(img, detector_type, param1, param2, param3), gt);
        std::cout << detector_name << " " << entry.path().filename().string()
                  << " | detectes=" << m.contours_detectes
                  << " reference=" << m.contours_reference
                  << " corrects=" << m.contours_corrects
                  << " FP=" << m.faux_positifs
                  << " FN=" << m.faux_negatifs
                  << " P=" << m.P << " TFP=" << m.TFP << " TFN=" << m.TFN << '\n';
        csv << detector_name << ',' << entry.path().filename().string() << ','
            << m.contours_detectes << ',' << m.contours_reference << ',' << m.contours_corrects << ','
            << m.faux_positifs << ',' << m.faux_negatifs << ',' << std::fixed << std::setprecision(6)
            << m.P << ',' << m.TFP << ',' << m.TFN << '\n';
        sum_P += m.P;
        sum_TFP += m.TFP;
        sum_TFN += m.TFN;
        ++count;
    }

    if (count > 0) {
        std::cout << "MOYENNE " << detector_name
                  << " | P=" << sum_P / count
                  << " TFP=" << sum_TFP / count
                  << " TFN=" << sum_TFN / count
                  << " | csv=" << csv_path << '\n';
    }
}

int main(int argc, char** argv) {
    std::string img_dir = fs::exists("dataset/images") ? "dataset/images" : "homework3/dataset/images";
    std::string gt_dir = fs::exists("dataset/ground_truth") ? "dataset/ground_truth" : "homework3/dataset/ground_truth";

    if ((argc == 5 || argc == 6) && std::string(argv[1]) == "--batch") {
        std::string detector_arg = argv[2];
        std::transform(detector_arg.begin(), detector_arg.end(), detector_arg.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        int detector_type = -1;
        if (detector_arg == "sobel" || detector_arg == "0") detector_type = 0;
        else if (detector_arg == "laplace" || detector_arg == "1") detector_type = 1;
        else if (detector_arg == "canny" || detector_arg == "2") detector_type = 2;
        else {
            std::cerr << "Detecteur invalide : " << argv[2]
                      << ". Utilisez sobel, laplace ou canny." << std::endl;
            return 1;
        }
        int param1 = std::stoi(argv[3]);
        int param2 = std::stoi(argv[4]);
        int param3 = argc == 6 ? std::stoi(argv[5]) : 3;
        if (param3 < 3 || param3 > 7 || param3 % 2 == 0) {
            std::cerr << "Le param3 doit etre impair et compris entre 3 et 7." << std::endl;
            return 1;
        }
        std::string detector_name = detector_type == 0 ? "Sobel" : (detector_type == 1 ? "Laplace" : "Canny");
        runBatch(detector_type, param1, param2, param3, img_dir, gt_dir,
                 "resultats_corriges_" + detector_name + ".csv");
        return 0;
    }

    // Fenêtre interactive pour le réglage des paramètres.
    int detector_type = 2; // 0: Sobel, 1: Laplace, 2: Canny
    int sobel_kernel = 3, sobel_threshold = 50;
    int laplace_kernel = 3, laplace_threshold = 30;
    int canny_low = 50, canny_high = 150, canny_aperture = 3;

    std::cout << "--- TP DETECTEURS DE CONTOURS ---" << std::endl;
    std::cout << "Choisissez un detecteur par son nom et ajustez ses parametres." << std::endl;
    std::cout << "Touches 1=Sobel, 2=Laplace, 3=Canny, b=analyse du lot, q=quitter." << std::endl;

    // Charger la première image disponible pour la mise au point interactive.
    cv::Mat sample;
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        sample = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!sample.empty()) break;
    }
    if (sample.empty()) {
        sample = cv::Mat::zeros(400, 400, CV_8UC1);
        cv::putText(sample, "Placez vos images dans dataset/", cv::Point(30, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);
    }

    std::string window_name = "TP Contours - Interface de demonstration";
    auto detector_name = [](int type) {
        return type == 0 ? std::string("Sobel") :
               (type == 1 ? std::string("Laplace") : std::string("Canny"));
    };
    auto createControls = [&]() {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("Choisir le detecteur (0=Sobel, 1=Laplace, 2=Canny)",
                           window_name, &detector_type, 2);
        if (detector_type == 0) {
            cv::createTrackbar("Sobel - taille du noyau (3, 5 ou 7)", window_name,
                               &sobel_kernel, 7);
            cv::createTrackbar("Sobel - seuil du gradient", window_name,
                               &sobel_threshold, 255);
        } else if (detector_type == 1) {
            cv::createTrackbar("Laplace - taille du noyau (3, 5 ou 7)", window_name,
                               &laplace_kernel, 7);
            cv::createTrackbar("Laplace - seuil de sensibilite", window_name,
                               &laplace_threshold, 255);
        } else {
            cv::createTrackbar("Canny - seuil bas (sensibilite)", window_name,
                               &canny_low, 255);
            cv::createTrackbar("Canny - seuil haut (confirmation)", window_name,
                               &canny_high, 255);
            cv::createTrackbar("Canny - taille ouverture (3, 5 ou 7)", window_name,
                               &canny_aperture, 7);
        }
    };
    createControls();

    int displayed_detector = detector_type;
    while (true) {
        if (detector_type != displayed_detector) {
            displayed_detector = detector_type;
            cv::destroyWindow(window_name);
            createControls();
        }

        int param1 = detector_type == 0 ? sobel_kernel :
                     (detector_type == 1 ? laplace_kernel : canny_low);
        int param2 = detector_type == 0 ? sobel_threshold :
                     (detector_type == 1 ? laplace_threshold : canny_high);
        int param3 = detector_type == 2 ? canny_aperture : 3;
        if (detector_type != 2) {
            // Un noyau de convolution doit rester impair et compris entre 1 et 7.
            param1 = std::max(1, std::min(7, param1 | 1));
        }
        cv::Mat display_edges = applyDetector(sample, detector_type,
                                               param1, param2, param3);

        cv::Mat source_display, edges_display, preview;
        cv::resize(sample, source_display, cv::Size(), 0.65, 0.65);
        cv::resize(display_edges, edges_display, source_display.size());
        cv::cvtColor(source_display, source_display, cv::COLOR_GRAY2BGR);
        cv::cvtColor(edges_display, edges_display, cv::COLOR_GRAY2BGR);
        cv::putText(source_display, "IMAGE ORIGINALE", cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(edges_display, "CONTOURS DETECTES", cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::hconcat(source_display, edges_display, preview);

        std::string selected_detector = detector_name(detector_type);
        std::string status = "DETECTEUR ACTIF : " + selected_detector;
        if (detector_type == 0)
            status += " | Noyau : " + std::to_string(param1) +
                      " | Seuil gradient : " + std::to_string(param2);
        else if (detector_type == 1)
            status += " | Noyau : " + std::to_string(param1) +
                      " | Seuil sensibilite : " + std::to_string(param2);
        else
            status += " | Seuil bas : " + std::to_string(param1) +
                      " | Seuil haut : " + std::to_string(param2) +
                      " | Ouverture : " + std::to_string(param3);
        cv::copyMakeBorder(preview, preview, 0, 78, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(30, 30, 30));
        cv::putText(preview, status, cv::Point(15, preview.rows - 49),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1);
        cv::putText(preview, "1 Sobel   |   2 Laplace   |   3 Canny   |   B : analyser les images   |   Q : quitter",
                    cv::Point(15, preview.rows - 19), cv::FONT_HERSHEY_SIMPLEX,
                    0.52, cv::Scalar(180, 220, 180), 1);
        cv::imshow(window_name, preview);
        
        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 27) break;

        // Les touches 1, 2 et 3 offrent un choix clair pendant la presentation.
        if (key == '1' || key == '2' || key == '3') {
            detector_type = key - '1';
        }
        
        if (key == 'b') { // Batch processing
            std::cout << "\n=== TRAITEMENT DU LOT D'IMAGES ===" << std::endl;
            std::cout << "Détecteur sélectionné: " << (detector_type == 0 ? "Sobel" : (detector_type == 1 ? "Laplace" : "Canny")) << std::endl;
            
            std::string selected_name = detector_name(detector_type);
            std::string csv_path = "resultats_contours_" + selected_name + ".csv";
            runBatch(detector_type, param1, param2, param3, img_dir, gt_dir, csv_path);
        }
    }

    return 0;
}