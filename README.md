# 🧠 INTEGRATED WELL SYNC

## 📌 Overview
This project presents a geoscience software developed to correlate stratigraphic layers by integrating multiple advanced computational and data-driven approaches, including:

* INPEFA–DTW (Integrated Prediction Error Filter Analysis with Dynamic Time Warping)
* MRGC (Multi-Resolution Graph-based Correlation)
* Machine Learning models: Random Forest and XGBoost
* Expert geological interpretation

The system is optimized for both CPU and GPU (CUDA) execution to enhance computational efficiency. It also features an interactive graphical user interface (GUI) that supports end-to-end workflows, confidence visualization, and exportable correlation results.
---

## 🚀 Features

* 🔗 Multi-method stratigraphic correlation (INPEFA–DTW, MRGC, ML-based)
* ⚡ CPU & GPU (CUDA) acceleration
* 🖥️ Interactive GUI for workflow execution
* 📊 Confidence visualization for interpretation reliability
* 📤 Exportable outputs for further analysis

---

## 🧩 Methodology

This work builds upon and extends existing open-source and academic contributions:
Modified implementation based on PyNPEFA
  👉 https://github.com/daeIy/PyNPEFA
Adaptation of Dynamic Time Warping correlation from
  👉 https://github.com/deepikaverma-geo/dtw-based-well-log-correlation
Algorithmic foundation inspired by L1 trend filtering:
  👉 https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

Enhancements include:

* Integration of multiple correlation techniques into a unified workflow
* Machine learning augmentation for pattern recognition
* Performance optimization using CPU/GPU acceleration
* Improved usability through GUI-based interaction

---

## ⚖️ License Compliance

This project includes modified code from:

- https://github.com/daeIy/PyNPEFA (GNU GPL v3)
- https://github.com/deepikaverma-geo/dtw-based-well-log-correlation (MIT)

Modifications and integrations have been performed to combine multiple correlation methods into a unified framework.

---

## 📚 Citation

If you use this software in academic work, please cite the following:

### Software

```
Fahriza Risma Annaza, 2026.
Geoscience Stratigraphic Correlation Tool:
INPEFA–DTW, MRGC, and Machine Learning Integration.
GitHub Repository.
```

### References

```
daeIy. PyNPEFA. GitHub repository.
https://github.com/daeIy/PyNPEFA

Deepika Verma. dtw-based-well-log-correlation. GitHub repository.
https://github.com/deepikaverma-geo/dtw-based-well-log-correlation

Boyd, S., et al. L1 Trend Filtering.
https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

Kang, Q. & Lu, L., 2020. Application of random forest algorithm in classification of logging lithology. Global Geology, 39(2), pp.398–405. Available at: https://doi.org/10.3969/j.issn.1004-5589.2020.02.014 

Ye, S.-J. & Rabiller, P., 2000. A new tool for electro-facies analysis: Multi-resolution graph-based clustering. In SPWLA 41st Annual Logging Symposium. Society of Petrophysicists and Well Log Analysts, pp.SPWLA-2000-PP. Available at: https://onepetro.org/SPWLAALS/proceedings-pdf/SPWLA-2000/SPWLA-2000/SPWLA-2000-PP/1914350/spwla-2000-pp.pdf


```

---

## 🙏 Acknowledgements

This project includes and builds upon the following works:

* PyNPEFA (GNU GPL v3 License)
* dtw-based-well-log-correlation (MIT License)
* L1 Trend Filtering (Stanford University)

All original authors retain their respective copyrights.

## ⚠️ Disclaimer

This software is provided "as is", without warranty of any kind. The authors are not responsible for any damages or misinterpretations resulting from its use.


## 🤝 Contributions

Contributions, issues, and feature requests are welcome!

## 📬 Contact

For academic collaboration or inquiries, please reach out via GitHub.

 
