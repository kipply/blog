+++
title = "On Interviewing, (System Design|Implementation|Debugging)"
date = 2022-05-9
weight = 2
+++

When it comes to engineers complaining about how they don't like interviewing, my top counterpoint is that being really good at interviewing makes you really good at interviews. Around a year ago I interviewed at Anthropic and didn't get an offer and six months later I did (and got other offers from companies that had in my opinion, harder interviews) with no canonical interview practice. I think a key thing that changed (which was not the most important one, but a substantial one) was designing and conducting interviews.

Through this post I'm going to describe my model of Cohehres (at the time I left, it was one of each problem with implementation as the phone screen)

### On Interviewing, Implementation



### On Interviewing, Debugging



### Evaluating Background Knowledge

I think in this interview, background knowledge (kubernetes, language models) is a big advantage and I want to limit the degree to which people without experience in the topics are disadvantaged. Some ways of doing that

- Ask lots of questions to get a good understand of their understanding, that will help you evaluate their reasoning and analysis skills based on their limited knowledge
- Solutions that would be a bad idea to execute do not necessarily indicate poor candidate performance
- Let the candidate make incorrect assumptions if you think that simplifies the problem

### Hint Guide

- Hints about solutions are good and encouraged
    - Coming up with solutions on the spot is really hard, I think if we hint them a solution, and they can subsequently explain the benefits and tradeoffs, that’s just as good.
    - eg if they start thinking about how requests take different times, and they realise “huh this would suck if you batch together a short request and a long request, that’s hard to solve” you can hint them towards solutions.
- Hints that identify problems for the candidate are sometimes bad
    - Given the previous example, if they notice that requests take different times and that batches will take the duration of the longest request in the batch (not everyone goes here, some people focus on other aspects) and they *don’t* realise that this is problematic for latency, that should be penalised.
        - this includes the comment hint pattern of “ok but what about this case”.
    - Again with the same example, it’s also the case that candidates might not know that a batch takes the duration of the longest request in the batch. I find sometimes candidates freeze a little after exploring one aspect of the problem, and don’t know which problem to tackle next. In this case, telling them “hey btw the batch takes the duration of the longest request in the batch, what issues might that cause?” is ok.
- Hints that add necessary information for the candidate are good and encouraged
    - Sometimes the candidate will explore down a solution path, and be missing some information needed to help them formulate a final solution
        - this is like revealing a constant, some detail about the setup, a specific kubernetes feature that exists, etc
    - Be a little careful with this, if they’re missing a significant amount of information you probably just want to guide them away from this path
- Hints should be penalised if poorly acted on
    - Some kinds of hints are good to give and should not cause penalty to the candidate, but if they struggle to reason through it or are resistant to accepting/exploring it that’s bad
