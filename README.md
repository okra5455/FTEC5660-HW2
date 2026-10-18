# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
The solution implements a multi-stage agentic AI workflow to verify CVs against LinkedIn and Facebook profiles using MCP tools. First, the agent extracts key claims from each CV, including the candidate’s name, professional title, location, employment history, education, and skills. It then searches LinkedIn using the candidate’s name and location, with an industry-based fallback when search results reach the limit without identifying a suitable match, and retrieves relevant profile details. Facebook is used as supplementary evidence to help confirm the candidate’s identity. Finally, the agent compares the extracted CV claims with the retrieved profile information, checking for discrepancies in employment details, education, and skills while accounting for acceptable wording differences and distinguishing missing information from unsupported claims. The workflow uses structured JSON outputs between stages, bounded concurrency, and retries to improve reliability. 

On the public test set, the solution achieved **100% accuracy (7/7)**.

| CV | Score | Correctness |
|---|---:|---|
| CV_1.pdf | 0.9500 | Correct |
| CV_2.pdf | 0.9500 | Correct |
| CV_3.pdf | 0.9500 | Correct |
| CV_4.pdf | 0.1500 | Correct |
| CV_5.pdf | 0.1500 | Correct |
| CV_6.pdf | 0.9500 | Correct |
| CV_7.pdf | 0.1500 | Correct |


### Task 2
The attack uses a visual overlay that the PDF text layer does not contain. We kept the education line as "BSc in Finance", which matches the profile, and placed a small image of the letter "M" over the "B". No hidden text, alt text, annotations, or metadata carry the false claim, so there is no secondary channel that could leak "MSc" into an extractor's output; the lie exists solely in pixels, and every text-visible value in the CV remains true to the target profile.

This is expected to work because all five verifiers, per the problem statement, load CVs with markitdown(enable_plugins=False), which is a text-only parser with no OCR. They compare "BSc" with "BSc" and find no difference on degree, school, field, or year, and the rest of the CV is fully true. Since what the verifiers read is clean, the result should stay above 0.5 on all three runs rather than depending on a weak model's mood. The only ways it fails are if a verifier secretly uses OCR or a layout-aware parser, or if the grader checks for a real lie using the same text-only loader instead of reading the page like a person. Both go against the stated setup.

