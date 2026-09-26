# Manuscript Addendum — Motivation and Rationale

This file contains paste-ready text that integrates the "Why this project is developed"
explanation into the CBP manuscript. Everything is written in the manuscript's existing
academic voice, uses only the existing references [1]–[15], and makes no claims of
original laboratory results.

---

## 1. NEW SUBSECTION — insert as "2.4" at the end of Section 2 (Importance of the Study)

--- BEGIN PASTE BLOCK ---

2.4 Motivation and Rationale for the Project

Problem statement. Spectroscopic instruments such as FTIR and Raman provide the most
reliable evidence of polymer identity [3], but they are expensive, require trained
operators and careful sample preparation, and are therefore inaccessible to many
educational laboratories and low-budget monitoring programmes. Manual microscopy, the
usual alternative, is slow, subjective, and difficult to standardize. This project is
developed to address that gap: it demonstrates that a low-cost, image-based
computer-vision pipeline can automate the first stage of microplastic analysis — the
detection, counting, and morphological classification of suspected particles — while
explicitly deferring chemical confirmation to spectroscopic methods [3], [4].

(a) The analytical gap. Polymer-level identification currently depends on FTIR or Raman
spectroscopy, whose equipment, sample preparation, and interpretation requirements limit
their use for rapid, large-scale screening [3]. Nile Red fluorescence improves optical
contrast at low cost, but remains a visualization technique rather than a chemical
identity test [4], [5]. Consequently, a large gap exists between the need for widespread
microplastic screening and the analytical resources available to perform it.

(b) Why manual analysis does not scale. Manual particle-by-particle examination has four
practical weaknesses. First, it is slow, often requiring substantial time per sample.
Second, it is subjective, since different analysts may count and classify the same
particles differently. Third, operator accuracy declines with fatigue during long
imaging sessions. Fourth, results are recorded inconsistently, which limits
reproducibility and comparison between samples. These weaknesses are precisely what
automated image analysis is designed to reduce [6].

(c) Why computer vision and machine learning are suitable. Counting objects and sorting
them by shape is a task well matched to computer vision. Segmentation, geometric
measurement (area, length, width, aspect ratio, circularity), and morphological
classification (fiber, fragment, bead) can be performed consistently and repeatably by
software, and the open-source Python ecosystem (OpenCV, NumPy, scikit-learn) makes the
software cost negligible once a basic microscope and camera are available [6], [7]. The
bottleneck of the workflow shifts from human inspection time to computation, which is
cheap.

(d) Screening, not confirmation. An ordinary RGB or fluorescence image cannot establish
the polymer identity of a particle; spectroscopic confirmation is required for that
purpose [3]. The project therefore deliberately claims only what image evidence can
support: automated detection of suspected microplastics and morphological classification.
In this design, the computational pipeline acts as a first-pass filter that identifies
potentially contaminated samples and prioritizes particles for FTIR or Raman analysis,
rather than presenting itself as a replacement for the analytical laboratory. The
honesty of this scope is itself part of the scientific design of the project.

(e) Role of the software prototype. In this project, the software prototype is not a
commercial product. It is the Computer Science deliverable that makes the chemistry
pipeline visible, testable, and reproducible. It operationalizes the workflow of
Section 5 — image upload, preprocessing, segmentation, classification, a results
dashboard, and export — demonstrates the interdisciplinary bridge between Engineering
Chemistry and Computer Science, and provides the structured results and figures used in
this report.

(f) Educational rationale. The project is scoped for a first-year team and intentionally
uses beginner-friendly tools (Python, OpenCV, scikit-learn, a web interface, and standard
data-processing libraries). It teaches image processing, feature extraction, classical
machine learning, software architecture, and data visualization without claiming
research-grade resources or instrumentation, which makes it an appropriate
course-based demonstration of chemistry–computing integration.

(g) A realistic path to larger systems. The same pipeline forms the basis of the future
extensions described in Section 11.2, including inexpensive microscope adaptations [10],
IoT-based monitoring [9], and citizen-science applications. All such deployments are
treated as future work that requires calibration, contamination control, and validation
before real environmental use.

In summary, microplastic detection is currently expensive, slow, and manual. This
project demonstrates, at an educational scale, that a low-cost and honest image-based
screening tool can automate the first step of that process — with chemical confirmation
remaining where it belongs, in the spectroscopy laboratory [3].

--- END PASTE BLOCK ---

---

## 2. SMALL EDITS TO EXISTING SECTIONS

### 2.1 Abstract — insert one sentence

Location: Abstract, after the sentence ending "...counting, sizing, and morphology
analysis." (the paragraph answering the central question).

--- BEGIN PASTE BLOCK ---
The motivation for this approach is practical: spectroscopic confirmation is costly and
not universally accessible, manual microscopy is slow and subjective, and automated
image analysis is well suited to the repetitive counting and morphological sorting tasks
that precede chemical confirmation.
--- END PASTE BLOCK ---

### 2.2 Table of Contents — add one line

Under "2. Importance of the Study", add:

--- BEGIN PASTE BLOCK ---
   2.4 Motivation and Rationale for the Project
--- END PASTE BLOCK ---

### 2.3 Section 4 (Objectives) — add Objective 7

Location: after Objective 6 (Educational Integration).

--- BEGIN PASTE BLOCK ---
7. Software Prototype Development: To develop a user-friendly software interface that
implements the image-based screening workflow, clearly communicates its screening-only
nature, and allows results to be reviewed and exported for further analysis.
--- END PASTE BLOCK ---

### 2.4 Section 5.1 — optional one-line workflow extension

Location: extend the overall theoretical workflow chain to include the software stage:

--- BEGIN PASTE BLOCK ---
... → Counting/Sizing → Screening Results and Export (Software Prototype) → FTIR/Raman Confirmation
--- END PASTE BLOCK ---

### 2.5 Section 11.1 (Conclusion) — insert one sentence

Location: after "The central contribution ... rather than the replacement of chemical
analysis by software."

--- BEGIN PASTE BLOCK ---
The motivation of the project — the cost, speed, and objectivity of the screening stage —
is matched by its stated limitation: the pipeline identifies suspected particles, and
polymer identity remains a matter for spectroscopic confirmation [3].
--- END PASTE BLOCK ---

---

## 3. INTEGRATION MAP (team note — not for submission)

Where each point of the original "why this project was developed" explanation landed:

| Original point                                  | Manuscript location                  |
|-------------------------------------------------|--------------------------------------|
| 1. Instruments are expensive; access gap        | 2.4 (a); reinforces 2.3, cites [3]   |
| 2. Manual analysis does not scale               | 2.4 (b); reinforces 2.3, cites [6]   |
| 3. CV + ML fit the task; near-zero software cost| 2.4 (c); cites [6], [7]              |
| 4. Screening vs confirmation honesty            | 2.4 (d); consistent with Abstract, 9.4, 10.1 |
| 5. Website = CSE deliverable bridging CH101/CSE | 2.4 (e) + Objective 7 + Section 6    |
| 6. Educational value for first-year team        | 2.4 (f); reinforces 8.3              |
| 7. Path to IoT / citizen science                | 2.4 (g); reinforces 11.2, cites [9], [10] |

No new references are required; all citations reuse the existing list [1]–[15], so the
submission checklist item "References: Fifteen references" remains valid.

---

## 4. CONSISTENCY NOTE (team note — not for submission)

Section 6.3 of the manuscript currently describes the user interface as Streamlit.
The actual software prototype built for this CBP is a React (frontend) + Python FastAPI
(backend) web application with a trained scikit-learn model (model.joblib) and a CSV
export. Consider rewriting 6.3 (or adding a 6.4) to describe the real architecture so
the manuscript matches the demonstrable prototype:

- 6.3 Frontend: React single-page interface (Home, Upload, Results dashboard, Model
  explanation, Limitations pages) with charting and CSV export.
- 6.4 Backend: FastAPI REST API exposing upload → OpenCV preprocessing → segmentation →
  feature extraction → scikit-learn classification → JSON results.
- 6.5 Deployment and screenshots mapping to report Figures 1–10.
