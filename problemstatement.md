3. Re:Learn: Adaptive Multimodal
Learning Environment
Backstory:
Most Learning Management Systems follow a fixed sequence of lectures, PDFs,
videos, assignments, and quizzes. Even adaptive systems often change only
question difficulty rather than how a concept is taught.
A wrong answer is usually treated as simply incorrect, even though it may
reveal a specific misconception. A student may know the equations but
misunderstand the concept, or apply a programming rule incorrectly. Showing
the correct solution does not necessarily fix the underlying misunderstanding.
Problem Statement:
Key Features:
Develop Re:Learn, an AI-powered learning system that trains a model to
analyze a learner's answers, working, or code and identify the underlying
misconception rather than simply marking the response as incorrect.
Teams must choose one domain: introductory programming, algebra, or
physics. They must build and evaluate a model capable of distinguishing
between different misconceptions that can produce similar incorrect answers.
The system should use the model's diagnosis to generate an intervention
targeted to the identified misconception.
After the intervention, the system must reassess the learner and determine
whether the diagnosed misconception has actually been resolved, rather than
assuming that a correct follow-up answer means learning has occurred.
Misconception Dataset: Build a dataset of correct responses, incorrect
responses, and their underlying misconceptions.
Misconception Model: Train and evaluate a model to classify or infer the
misconception behind a learner's response.
Misconception Differentiation: Distinguish between different misconceptions
that produce similar mistakes.
Adaptive Intervention: Generate or select an intervention based on the
model's diagnosis.
Resolution Assessment: Determine whether the learner's underlying
misconception has been resolved after intervention.
Learner Model: Track recurring misconceptions and demonstrated
understanding across attempts.
Model Evaluation: Evaluate diagnosis accuracy and performance on
responses or misconceptions not seen during training