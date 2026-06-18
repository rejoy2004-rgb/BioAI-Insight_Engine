import json
import os
import random

def generate_papers():
    random.seed(42)
    
    topics = [
        "Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer", 
        "Pancreatic Cancer", "Ovarian Cancer", "Melanoma", "Leukemia", "Glioblastoma"
    ]
    
    technologies = [
        "Deep Learning Models", "Convolutional Neural Networks", "Transformer Architectures",
        "Liquid Biopsies", "Single-Cell RNA Sequencing", "CRISPR-Cas9 Gene Editing",
        "CAR-T Cell Immunotherapy", "Radiomics Feature Extraction", "Multi-Omics Integration",
        "Next-Generation Sequencing (NGS)", "Supervised Machine Learning Pipelines"
    ]
    
    focus_areas = [
        "Early Detection and Screening", "Survival Rate Prediction", "Drug Resistance Pathways",
        "Metastasis Modeling", "Tumor Microenvironment Analysis", "Immunotherapy Response Prediction",
        "Biomarker Discovery", "Histopathology Image Segmentation", "Clinical Trial Optimization"
    ]
    
    verbs = [
        "improves diagnostic accuracy of", "reveals novel therapeutic targets for",
        "predicts patient response during", "enhances early classification of",
        "characterizes tumor heterogeneity in", "identifies genetic risk factors for",
        "accelerates biomarker profiling in", "optimizes personalized treatment for"
    ]
    
    outcomes = [
        "achieving a sensitivity of 97.2% and reducing false positives by 30%.",
        "yielding a 15% improvement in overall survival prediction accuracy.",
        "uncovering key microenvironmental factors associated with chemotherapy resistance.",
        "demonstrating significant clinical utility in a retrospective cohort of 1,200 patients.",
        "reducing clinical turnaround times from 5 days to under 3 hours.",
        "providing explainable prediction pathways for oncological decision support."
    ]
    
    journals = [
        "Nature Cancer", "The Lancet Oncology", "Journal of Clinical Oncology",
        "Cancer Research", "Clinical Cancer Research", "Bioinformatics",
        "IEEE Transactions on Medical Imaging", "Oncogene", "JAMA Oncology",
        "NPJ Breast Cancer", "Frontiers in Oncology"
    ]
    
    authors_pool = [
        "A. Patel", "S. Jenkins", "M. Dubois", "K. Takahashi", "E. Rostova",
        "J. Smith", "C. Martinez", "L. Zhang", "R. Gupta", "H. Muller",
        "O. Nielsen", "T. Al-Fayed", "P. Kowalski", "Y. Tanaka", "E. Johnson"
    ]
    
    papers = []
    
    # Generate 510 unique papers
    for i in range(510):
        topic = random.choice(topics)
        tech = random.choice(technologies)
        focus = random.choice(focus_areas)
        verb = random.choice(verbs)
        outcome = random.choice(outcomes)
        
        title = f"{tech} for {focus} in {topic} Patients"
        
        # Adjust some titles for natural diversity
        if i % 3 == 1:
            title = f"How {tech} {verb} {topic}"
        elif i % 3 == 2:
            title = f"Clinical Utility of {tech} in {topic} {focus}"
            
        authors_count = random.randint(2, 5)
        authors = ", ".join(random.sample(authors_pool, authors_count))
        
        journal = random.choice(journals)
        year = random.randint(2020, 2026)
        
        # Build a structured abstract
        abstract = (
            f"Background: {focus} remains a critical challenge in managing {topic}. "
            f"Here, we propose a novel approach leveraging {tech} to address this gap. "
            f"Methods: We evaluated our framework on a comprehensive cohort of clinical samples and patient profiles. "
            f"Results: Our findings indicate that integrating {tech} {verb} {topic}, {outcome} "
            f"Conclusion: This study demonstrates that {tech} represents a viable clinical diagnostic tool for {topic} management."
        )
        
        papers.append({
            "paper_id": i + 1,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "abstract": abstract
        })
        
    os.makedirs("data", exist_ok=True)
    with open("data/papers_corpus.json", "w") as f:
        json.dump(papers, f, indent=4)
        
    print(f"Generated {len(papers)} papers in data/papers_corpus.json")

if __name__ == "__main__":
    generate_papers()
