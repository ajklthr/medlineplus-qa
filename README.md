# Medical Q&A System - Retrieval Augmented Generation (RAG)

## Overview

This open-source Medical Q&A System leverages **Retrieval Augmented Generation (RAG)** to deliver accurate and trustworthy responses from credible medical sources, ensuring patient safety and trust.

Healthcare organizations are increasingly exploring the use of **Generative AI (GenAI)** to enhance patient engagement via interactive, multi-round Q&A systems. However, concerns surrounding the probabilistic nature of token generation in **Large Language Models (LLMs)**—including hallucinations and inaccurate information—have raised significant patient safety and trust issues, leading to hesitancy in adopting such solutions.

The goal of this project is to develop a reliable, scalable multi-round Q&A system grounded in credible sources, either from the healthcare organization itself or reputable public sources like **MedlinePlus**. The system will incorporate robust guardrails to ensure patient safety and build trust while being scalable for healthcare organizations. The final solution is a fully functional **GenAI-driven Q&A system**, powered by **Llama 3.1**, that enhances patient engagement while addressing safety concerns.

## Supported Sources (as of June 7, 2024)

- **MedlinePlus**: A trusted online health information resource for patients and families. It is a service of the **National Library of Medicine (NLM)**, the world’s largest medical library, part of the **National Institutes of Health (NIH)**.

## Model

- **Llama 3.1** serves as the foundational Large Language Model (LLM) for generating responses in the multi-round Q&A system.

## Tech Stack

- **Kubeflow**: Manages pipelines for tasks like index creation and model fine-tuning.
- **KServe + VLLM**: Ensures efficient and scalable serving of the Llama 3.1 model.
- **Hybrid Search with FAISS**: Builds and queries vector indices for fast, reliable information retrieval.
- **Nemo Guardrails**: Implements safety mechanisms to ensure the AI delivers reliable outputs.
- **Redis**: Provides semantic caching to optimize response times and minimize unnecessary computations.
- **Kubernetes (K8s)**: Manages and deploys the RAG application in a containerized, scalable environment.
- **OpenTelemetry & Grafana Cloud**: Offers observability for monitoring and improving system performance.
- **RAGAS**: Evaluates the performance of the Retrieval Augmented Generation system.

## Project Goals

This system is designed to:

- Provide high-quality, multi-round Q&A interactions for patients.
- Leverage credible sources to deliver safe and trustworthy responses.
- Scale efficiently for deployment within healthcare organizations.
- Address both patient engagement and safety concerns through a robust GenAI-driven solution.
