# AI Ethics Assignment: Theoretical Analysis

## Part 1: Theoretical Understanding

### 1. Short Answer Questions

**Q1: Define algorithmic bias and provide two examples of how it manifests in AI systems.**

Algorithmic bias occurs when an AI system produces systematically prejudiced results due to erroneous assumptions in the machine learning process. This typically stems from biased training data or flawed algorithms that reflect and amplify existing societal biases.

Examples:
1. **Gender Bias in Hiring**: Amazon's recruiting AI system was found to be biased against female candidates because it was trained on resumes submitted over a 10-year period, which were predominantly from male applicants, causing it to downgrade resumes containing words like "women's" or the names of women's colleges.

2. **Racial Bias in Facial Recognition**: Many commercial facial recognition systems have shown higher error rates for people with darker skin tones, particularly women of color, due to underrepresentation in training data and failure to account for diverse facial features.

**Q2: Explain the difference between transparency and explainability in AI. Why are both important?**

**Transparency** refers to the openness about how an AI system is developed, what data it uses, and its intended purpose. It's about making the AI's operations and decision-making processes visible and understandable to stakeholders.

**Explainability** is the ability to explain, in human-understandable terms, how an AI system arrived at a particular decision or prediction. It focuses on making the reasoning behind AI outputs clear and interpretable.

**Importance:**
- **Transparency** builds trust with users and stakeholders by revealing how systems work and what data they use.
- **Explainability** ensures that decisions can be understood and challenged, which is crucial for accountability and fairness.
- Together, they enable better governance, help identify and mitigate biases, and ensure compliance with regulations like GDPR.

**Q3: How does GDPR (General Data Protection Regulation) impact AI development in the EU?**

GDPR impacts AI development in the EU in several key ways:

1. **Right to Explanation**: Individuals have the right to obtain meaningful information about the logic involved in automated decision-making that significantly affects them.

2. **Data Protection by Design**: AI systems must incorporate data protection measures from the initial design stage, including data minimization and purpose limitation.

3. **Consent and Control**: Explicit consent is required for processing personal data, and individuals have the right to withdraw consent at any time.

4. **Right to Human Intervention**: Individuals can request human intervention, express their point of view, and contest automated decisions.

5. **Data Subject Rights**: Includes rights to access, rectification, erasure (right to be forgotten), and data portability.

6. **Impact Assessments**: Required for high-risk processing, including certain AI applications, to assess risks to individuals' rights and freedoms.

These requirements encourage the development of more transparent, accountable, and privacy-preserving AI systems in the EU.

### 2. Ethical Principles Matching

1. **B) Non-maleficence**: Ensuring AI does not harm individuals or society.
2. **C) Autonomy**: Respecting users' right to control their data and decisions.
3. **D) Sustainability**: Designing AI to be environmentally friendly.
4. **A) Justice**: Fair distribution of AI benefits and risks.

## Part 2: Case Study Analysis

### Case 1: Biased Hiring Tool

**Source of Bias:**
The bias in Amazon's AI recruiting tool primarily stemmed from:
1. **Training Data Bias**: The model was trained on resumes submitted to Amazon over a 10-year period, which were predominantly from male applicants, reflecting historical hiring patterns in the tech industry.
2. **Feature Selection**: The system learned to associate certain terms (like "women's chess club") with female candidates and penalized them, as these terms were less common in the successful resumes it was trained on.
3. **Feedback Loop**: The system reinforced existing biases by learning from past hiring decisions that may have been influenced by human biases.

**Proposed Fixes:**
1. **Debiasing Training Data**:
   - Remove gender-identifying information from resumes before processing
   - Use synthetic data to balance underrepresented groups
   - Implement data augmentation techniques to ensure fair representation

2. **Algorithmic Fairness Interventions**:
   - Apply pre-processing techniques to remove bias from training data
   - Use in-processing methods that incorporate fairness constraints during model training
   - Implement post-processing adjustments to ensure fair outcomes

3. **Human-in-the-Loop System**:
   - Maintain human oversight for final hiring decisions
   - Implement regular audits of the system's decisions
   - Create a feedback mechanism to continuously improve the system

**Fairness Metrics for Evaluation:**
1. **Demographic Parity**: Ensure similar selection rates across gender groups
2. **Equal Opportunity**: Equal true positive rates across groups
3. **Predictive Parity**: Similar precision across groups
4. **Disparate Impact Ratio**: Ratio of selection rates between protected and non-protected groups (should be close to 1)
5. **Average Odds Difference**: Average of the difference in false positive and true positive rates between groups

### Case 2: Facial Recognition in Policing

**Ethical Risks:**
1. **Wrongful Arrests**: Higher misidentification rates for minorities can lead to false accusations and arrests, with severe personal and legal consequences.
2. **Privacy Violations**: Mass surveillance capabilities can infringe on civil liberties and create a chilling effect on free assembly and expression.
3. **Reinforcement of Systemic Bias**: Over-policing of minority communities can be exacerbated if systems are deployed more heavily in these areas.
4. **Lack of Accountability**: Difficulty in challenging or understanding automated decisions can undermine due process.
5. **Function Creep**: Initial limited deployments may expand beyond their original scope without proper oversight.

**Policy Recommendations:**
1. **Pre-Deployment Requirements**:
   - Mandate rigorous testing for bias across demographic groups
   - Require transparency about accuracy rates and limitations
   - Establish clear use cases and restrictions

2. **Ongoing Oversight**:
   - Regular third-party audits of system performance
   - Public reporting of usage statistics and outcomes
   - Establish review boards with diverse stakeholders

3. **Operational Safeguards**:
   - Prohibit sole reliance on facial recognition for arrests
   - Require human verification of all matches
   - Implement strong evidentiary standards for matches

4. **Legal Protections**:
   - Right to legal recourse for individuals affected by errors
   - Strict data retention limits
   - Prohibition of mass surveillance applications

5. **Community Engagement**:
   - Public consultation before deployment
   - Transparency about where and how the technology is used
   - Mechanisms for public feedback and complaint

6. **Sunset Provisions**:
   - Regular re-evaluation of system effectiveness and impact
   - Automatic expiration of deployment authorizations without renewal
   - Clear criteria for system decommissioning

## Part 4: Ethical Reflection

[Your personal reflection on how you would ensure ethical AI principles in a past or future project would go here. Discuss specific steps you would take to ensure fairness, transparency, and accountability in your work.]
