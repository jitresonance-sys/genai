INTELLIGENT EV CHARGING
DEMAND PREDICTION
& AGENTIC INFRASTRUCTURE
PLANNING

From Usage Analytics to Autonomous Grid & Station Planning

PROJECT OVERVIEW
This project focuses on building an AI-driven analytics system for electric vehicle
(EV) infrastructure planning.
In Milestone 1, classical machine learning techniques are used to predict EV
charging demand using historical charging station usage, time, and location data.
In Milestone 2, the system evolves into an agentic AI assistant that reasons about
charging demand patterns, retrieves infrastructure planning guidelines, and
generates structured recommendations.
The project addresses a real-world sustainability problem and demonstrates applied
machine learning, agentic AI design, and system deployment.

CONSTRAINTS & REQUIREMENTS

TEAM SIZE
3–4 Students

API BUDGET
Free Tier Only

FRAMEWORK
LangGraph (Recommended)

HOSTING
Mandatory

APPROVED TECHNOLOGY STACK

LLMS (MILESTONE 2)
Open-source models
Free-tier APIs

AGENT FRAMEWORK
LangGraph (Recommended)
Chroma / FAISS (RAG)

ML & ANALYTICS (MILESTONE 1)
Scikit-Learn (Pipelines)
Regression/Classification
Feature Engineering

UI FRAMEWORK
Streamlit (Recommended)
Gradio

HOSTING PLATFORM (MANDATORY)
Hugging Face Spaces
Streamlit Community Cloud
Render (Free Tier)

WARNING: Localhost-only demonstrations will not be accepted.

MILESTONE 1: ML-BASED EV CHARGING DEMAND
PREDICTION

MID-SEM SUBMISSION

Objective: Predict EV charging demand at stations using historical usage data.
Focus on cleaning real-world data and building robust predictive models
without LLMs.
Functional Requirements:
Accept charging session and location data.
Perform data preprocessing and feature engineering.
Predict charging demand (Regression/Classification).
Display demand usage trends via user interface.

TECHNICAL REQUIREMENTS (ML)

INPUTS & OUTPUTS

MID-SEM DELIVERABLES
■ Preprocessing: Time-series features, Cleaning.
■ Features: Location, Time of Day, Usage history.
■ Models: Random Forest, Gradient Boosting.
■ Evaluation: MAE, RMSE, R-Squared.

■ Input: Charging session CSV data.
■ Output: Demand Prediction.
■ Metrics: Trend Analysis.

MILESTONE 2: AGENTIC EV INFRASTRUCTURE
PLANNING

END-SEM SUBMISSION

Objective: Evolves into an agentic AI assistant that reasons about charging
demand patterns and generates optimized infrastructure and scheduling
recommendations.
Functional Requirements:
Analyze demand for high-load locations.
Retrieve infrastructure planning guidelines.
Generate optimal charger placement recommendations.
Display scheduling optimization insights.

TECHNICAL REQUIREMENTS (AGENTIC)

STRUCTURED OUTPUT
Problem understanding & Domain
description.
■

■ Input–output specification.

■ System architecture diagram.
■ Working local application with UI.
■ Model evaluation report.

■ Framework: LangGraph (Workflow & State).
■ RAG: Planning Guidelines (Chroma/FAISS).
■ State: Explicit state management.
■ Logic: Optimization reasoning.

■ Analysis: Charging Demand Summary.
■ Locate: High-load Location ID.

END-SEM DELIVERABLES

Final Artifacts: Hosted Link, GitHub Repo, Demo Video.

EVALUATION CRITERIA

PHASE WEIGHT CRITERIA

Mid-Sem
(Milestone 1)

25%

Correct ML pipeline & Demand Prediction
Quality of Data Preprocessing & Evaluation
UI Usability & Domain Understanding

End-Sem
(Milestone 2)

30%

Quality of Agent Reasoning & Planning
Retrieval Integration (RAG) & State
Deployment Success & Video Clarity
Infrastructure Optimization Logic