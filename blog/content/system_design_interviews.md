+++
title = "System Design Interview Meta"
date = 2022-08-25
weight = 2
+++

This is a slightly unorganized document containing thinking about system design interviews. It's not at all interview advice but it may be helpful for understanding what a system design interview is about. It would be more helpful for a team thinking about how to conduct and design the interviews.

### skills

The reason there are different kinds of interviews is to test for different skills. It’s worth thinking about what skills a system design interview test for.

The list is something like:

- background knowledge
    - all interview questions will call for knowledge of varying types, but system design will also call on [tacit knowledge](https://nintil.com/scaling-tacit-knowledge/#tacit-knowledge)
- ability to adapt for constraints or new tools
    - in system design, you’ll get a lot of “ok but we can’t compromise x”, "let's say we can't use x" and “how can you use z to help you?” as subproblems.
- planning
    - the planning here differs from say implementation problems since it'll be less about laying out the steps and more about preparing for adaptability (the steps become a graph) and predicting circumstances.
- communication
    - you want to accurate communicate your solution, the reasoning for applying that solution and how you plan to execute it. system design tests the latter two parts more extensively.
- problem solving
    - main difference from implementation questions is that the solved state is less defined. that relates to the fact that he process will involve more exploration (sampling of tools, edge cases, use cases, contstraints to consider, etc) and creativity.

### background knowledge

System design interviews almost always cover and expect background knowledge. Applying the knowledge demonstrates ability to pattern match problems, to reflect on solutions they have tried and the past. This is the main reason system design interviews are better for more senior / experienced people, ass they capture how valuable their experience is.

Domain expertise allows a big advantage, and it may be hard to evaluate skills (say, their ability to think critically about solutions they’ve attempted) in isolation. Being able to isolate the skills improves reasoning about performance and improves calibration.

One could cleanly separate it out by just asking them questions about their domain, but I find it’s possible to isolate in the system design interview.

> As an aside, I'm quite optimistic about the idea of quizzes as interviews. Not like "fill out a form" but someone will talk to you with a checklist of things they'd like you to know.

Some conducting guidelines about this.

- Start by asking lots of questions to get a good understanding of their background to help you more accurately evaluate their reasoning and analysis skills and also guide them more effectively through the problem solving
- Solutions that would be a bad idea to execute do not necessarily indicate poor candidate performance. Sometimes missing information can result in a bad idea, but perhaps the ideation and evaluation process was still impressive given what they knew
- Let the candidate make incorrect assumptions if you think that simplifies the problem

I think these conducting guidelines are quite common, and are the reason why I find lots of people are surprised about their evaluation on system design questions where they can accurately evaluate their own performance on implementation questions. It’s very easy for the candidate to not know that there might be a better solution given more knowledge, especially of the tacit kind.

### guidance

System design interviews should be very interactive, as they tend to be open-ended so candidates should be asking questions like “well we can perform a trade off between latency and throughput here, which do we care about more?”. They're also harder to conduct for this reason, as you may need to guide the candidate to cover grounds for evaluation.

Here are some useful prompts to capture skills

- Give the candidate an additional piece of information, say a constraint or a tool they have access to and evaluate how well they update their thinking based on the new information.
- Engage in discussion for questions the candidate asks. If they asked the question about latency and throughput I proposed above, don’t just give an absolute answer but discuss which one makes more sense for whatever is being worked on
- Talk about whether they’ve solved any similar (sub)problems, and how they’re applying that knowledge to the problem at hand. A good intuition is excellent, but being able to identify where that intuition comes from is even better.
- Ask questions to check their understanding or to have them develop it. A common failure mode is for the candidate to give a big blanket solution like “oh we can autoscale with kubernetes” and then not go into it further and either that part gets lost or even the interview could end too early, without enough signal.

All of these guidance actions are specifically not hints – they are not necessarily things that are done when struggle is observed, rather they are ways to stay engaged with the process and get more signal from the interview.

### hint guide

Thinking about hinting is really important for calibration but also useful for having a pleasant interview. It's not always the case that hints are given for poor performance. Hints are really helpful to help direct the candidate down a certain path where you think you can get a more thorough evaluation or to just unstuck them on things you don't expect them to figure out in the interview.

- Hints that direct people to solutions are good, and should be only lightly penalised
    - If a solution is hinted, and they can quickly understand the proposed solution and explain the benefits and tradeoffs, that’s nearly as good. Ideation is really hard in an interview setting, and many of the best systems solutions come passively from prolonged experience with problem.
- Hints that identify problems for the candidate should usually be penalised
    - Not identifying a problem is evidence of blind-spots or poor reasoning skills. Sometimes they lack information to be certain it’s a problem, but it’s often the case that they should be at least capable of exploring whether it’s a problem.
    - Exceptions for things like small edge cases that they immediately understand.
- Hints that add necessary information for the candidate should not be penalised
    - Sometimes the candidate will explore down a solution path, and be missing some information needed to help them formulate a final solution. This can be revealing constants or letting them know about the existence of a feature.
    - Need to be a little careful with this, because if they lack too much knowledge it may make more sense to direct them down a different solution path
- Hints should be penalised if poorly acted on
    - anything from not acknowledging the hint to failing to infer the correct updates from it.

### problem specification

The list of properties a good system design question should have is uninteresting. Should be a realistic problem that gives space for candidates to explore the complexities of system design and apply their various experiences. Not too open-ended such that the field of answering is too large and not too closed such that the solution is too narrow. Shouldn’t be about link shortening.

It actually turned out to be that coming up with the premise for a good question wasn't that hard, but specifying it well kind of was. Here are the things I would recommend for that:

- Keep a list of possible solution paths
    - It’s really important for interviewers to be very familiar with the problem. It's good for preparing interviewers and helps calibrate. Calibrating by comparing against past evaluations is more effective when the past evaluation is closer to the one at hand
- Keep a list of questions that the candidate might ask
    - This is also a list of questions that candidates *should* ask
- Keep a list of possible constants / specifications the candidate may ask for / need
    - this is common for doing some napkin math.
- Prepare a clear list of goals to optimise for
- Problem spec should include a use case example
    - This is really helpful for the candidate to use as reference when explaining solutions. Not to say that candidate shouldn’t need to come up with other possible use cases
- Create a rubric of specific skills you’re trying to capture, with examples specific to the question
- Preset notes about common hints to give and questions to ask
    - this is useful for calibration, consistency and for interviewer education

The specs I maintained grew quickly as I conducted the question for the first few times, it's probably not necessary to have the whole document the first time, especially as some are more important for scaling number of interviewers.

---

Before interviewing for my current job I had actually problem set more system design interviews than I had taken (which is also the time where I developed this thinking, doing more interviews didn't give me any new ideas). Not really sure how correct this thinking is but it's at least the product of a lot of first principles thinking and iteration so hopefully insightful!