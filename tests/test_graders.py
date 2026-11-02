from datetime import date

import pytest
from conftest import GT, jl

from rl_env_use.graders import grade_answer, norm, number, numbers, parse_date

CASES = jl(GT / 'grader_cases.jsonl')


@pytest.mark.parametrize('case', CASES, ids=[f'{c["task"]["answer_type"]}:{c["why"]}' for c in CASES])
def test_grader_ground_truth(case):
	assert grade_answer(case['task'], case['pred']) is case['label'], case['why']


def test_ground_truth_covers_every_answer_type_both_ways():
	seen = {(c['task']['answer_type'], c['label']) for c in CASES}
	assert seen == {(t, lab) for t in ('number', 'date', 'exact', 'contains', 'set') for lab in (True, False)}


def test_web_ground_truth_oracle_passes():
	for t in jl(GT / 'web_tasks.jsonl'):
		assert grade_answer(t, t['answer']), t['id']


@pytest.mark.parametrize('s,expected', [('1,234.5', [1234.5]), ('3 5', [3, 5]), ('1.2M users', [1.2e6]), ('-0.5', [-0.5]), ('v2.1', [2.1]), ('none', [])])
def test_numbers(s, expected):
	assert numbers(s) == expected


@pytest.mark.parametrize('s', ['2023-07-25', '25 Jul 2023', 'July 25, 2023', '25.07.2023', 'Tue, 25 Jul 2023 22:59:32 UTC', '25th July 2023'])
def test_parse_date(s):
	assert parse_date(s) == date(2023, 7, 25)


def test_norm():
	assert norm('  The  Zürich-Pile. ') == 'zurich pile'
	assert norm('v3.11') == 'v3.11'


def test_bad_specs_raise():
	with pytest.raises(ValueError):
		grade_answer({'answer_type': 'regex', 'answer': 'x'}, 'x')
	with pytest.raises(ValueError):
		grade_answer({'answer_type': 'number', 'answer': 'many'}, '3')


def test_known_limitation_contains_colon_hedge():
	# `contains` is for free-text answers; "Paris: Lyon" is indistinguishable from "Answer: Lyon" without brittle word lists.
	assert grade_answer({'answer_type': 'contains', 'answer': 'Lyon'}, 'Paris: Lyon')


def test_known_limitation_spelled_out_numbers():
	# Documented limitation (policies are asked for digits): words are not parsed, so this fails even though a human would accept it.
	assert not grade_answer({'answer_type': 'number', 'answer': '812'}, 'eight hundred twelve')


def test_trace_metrics():
	from rl_env_use.graders import grounded, max_repeats, page_key

	assert page_key('https://arxiv.org/html/2307.13854v4#S5') == page_key('http://www.arxiv.org/abs/2307.13854') == 'arxiv:2307.13854'
	tr = [
		{'role': 'assistant', 'content': '[step 0 @ https://duckduckgo.com/?q=webvoyager]\nnext_goal: x'},
		{'role': 'tool_result', 'content': 'snippet: Mind2Web 137 websites ... WebVoyager 643 tasks'},
		{'role': 'assistant', 'content': '[step 1 @ https://github.com/MinorJerry/WebVoyager]'},
	]
	wv, m2w = {'url': 'https://github.com/MinorJerry/WebVoyager', 'value': '643'}, {'url': 'https://arxiv.org/abs/2306.06070', 'value': '137'}
	assert grounded([wv], tr)
	assert not grounded([wv], tr[:2])  # snippet only
	assert not grounded([m2w], [{'role': 'assistant', 'content': '[step 0 @ https://arxiv.org/]'}] + tr[:2])  # snippet read on duckduckgo, not on arxiv
	assert not grounded([wv], [{'role': 'assistant', 'content': '[step 0 @ https://github.com]'}])  # root is not the source
	assert not grounded([wv, m2w], tr)  # multi-source: every fact must be read at its source
	listing = [
		{'role': 'assistant', 'content': '[step 1 @ https://arxiv.org/search/?query=mind2web]'},
		{'role': 'tool_call', 'name': 'extract', 'content': '{}'},
		{'role': 'tool_result', 'content': 'Mind2Web ... collected from 137 websites'},
	]
	assert grounded([wv, m2w], tr + listing)  # same-site listing page that shows the value counts
	assert not grounded([m2w], [{'role': 'assistant', 'content': '[step 0 @ https://arxiv.org/abs/2307.13855]'}])
	assert not grounded([], tr)
	opened_not_viewed = [{'role': 'tool_call', 'name': 'navigate', 'content': '{"url": "https://arxiv.org/abs/2306.06070", "new_tab": true}'}]
	assert not grounded([m2w], opened_not_viewed)
	dated = [
		{'role': 'assistant', 'content': '[step 0 @ https://arxiv.org/search/?q=x]'},
		{'role': 'tool_call', 'name': 'extract', 'content': '{}'},
		{'role': 'tool_result', 'content': 'Submitted 9 June, 2023; originally announced June 2023.'},
	]
	assert grounded([{'url': 'https://arxiv.org/abs/2306.06070', 'value': '9 Jun 2023'}], dated)
	assert max_repeats([{'role': 'tool_call', 'name': 'evaluate', 'content': 'x'}] * 13) == 13 and max_repeats([]) == 0


def test_grounded_resists_self_grounding_and_url_tricks():
	"""Exploits found by the adversarial review: the policy's own outputs and look-alike URLs must never ground it."""
	from rl_env_use.graders import grounded

	ev = [{'url': 'https://arxiv.org/abs/2401.12345', 'value': '42.7'}]
	on_root = {'role': 'assistant', 'content': '[step 0 @ https://arxiv.org/]'}
	assert not grounded(ev, [on_root, {'role': 'tool_call', 'name': 'done', 'content': '{}'}, {'role': 'tool_result', 'content': '{"answer": "42.7"}'}])
	assert not grounded(ev, [on_root, {'role': 'tool_call', 'name': 'input', 'content': '{}'}, {'role': 'tool_result', 'content': "Input '42.7' into element 3"}])
	repo = [{'url': 'https://github.com/owner/repo', 'value': '643'}]
	assert not grounded(repo, [{'role': 'assistant', 'content': '[step 0 @ https://github.com/owner/repo/issues/new?title=643]'}])
	assert not grounded(repo, [{'role': 'assistant', 'content': '[step 0 @ https://github.com/owner/repo-evil]'}])
	assert not grounded([{'url': 'https://python.org', 'value': '3.12'}], [{'role': 'assistant', 'content': '[step 0 @ https://python.org/about]'}])
	assert not grounded(repo, [{'role': 'assistant', 'content': '[step 0 @ https://github.com/owner/repo/issues/new]'}])
	assert not grounded(repo, [{'role': 'assistant', 'content': '[step 0 @ https://github.com/owner/repo/search?q=643]'}])
	hf = [{'url': 'https://huggingface.co/datasets/McGill-NLP/WebLINX', 'value': '8'}]
	assert grounded(hf, [{'role': 'assistant', 'content': '[step 0 @ https://huggingface.co/datasets/McGill-NLP/WebLINX/viewer]'}])  # plain sub-page view
	assert grounded(
		repo,
		[
			{'role': 'assistant', 'content': '[step 0 @ https://github.com/owner/repo/blob/main/README.md]'},
			{'role': 'tool_call', 'name': 'extract', 'content': '{}'},
			{'role': 'tool_result', 'content': 'The benchmark has 643 tasks'},
		],
	)


def test_digits_inside_identifiers_are_not_numbers():
	from rl_env_use.graders import numbers

	assert numbers('SCN5A') == [] and numbers('TP53 in gnomAD v4') == [4.0] and numbers('rs2017481') == []
	assert numbers('5 residues, 1.2k, 12 kDa') == [5.0, 1200.0, 12.0]
	assert not grade_answer({'answer': '5', 'answer_type': 'number'}, 'SCN5A')


def test_position_ranges_match_in_any_wording():
	t = {'answer': '77-149', 'answer_type': 'exact'}
	assert grade_answer(t, 'Start: 77, End: 149') and grade_answer(t, '77 to 149') and grade_answer(t, '77–149')
	assert not grade_answer(t, '77-150') and not grade_answer(t, '149-77') and not grade_answer(t, '77')


def test_restated_units_match_the_digits_the_prompt_asked_for():
	t = {'answer_type': 'number', 'answer': '3169', 'prompt': 'Compute net debt (in thousands of dollars).'}
	assert grade_answer(t, '$3,169 thousand') and grade_answer(t, '3,169') and not grade_answer(t, '$3,169 million') and not grade_answer(t, '3,170 thousand')
	assert not grade_answer(t | {'prompt': 'How many websites?'}, '3,169 thousand')  # no unit in the prompt: the scale word counts
	assert grade_answer(t | {'answer': '1.9', 'prompt': 'Q3 net loss in millions?'}, '$1.9 million') and number('3.2 million', '3,200,000')
	assert not number('1521.33, 5.07', '5.07')  # two different numbers is still hedging
	assert grade_answer(t | {'prompt': 'Express it in €1,000s.'}, '€3,169 thousand') and grade_answer(t | {'prompt': 'Net debt (in $000s)?'}, '3,169 thousand')


@pytest.mark.parametrize('pred', ['-$4.50 per share', '($4.50)', '$(4.50)', '–$4.50', '- $4.50', '-4.50'])
def test_finance_negatives_parse_signed(pred):
	assert grade_answer({'answer_type': 'number', 'answer': '-4.50', 'abs_tol': 0.005}, pred)
	assert not grade_answer({'answer_type': 'number', 'answer': '4.50', 'abs_tol': 0.005}, pred)


def test_ranges_and_plain_parentheses_stay_unsigned():
	from rl_env_use.graders import numbers

	assert numbers('2021–$5') == [2021.0, 5.0] and numbers('(n=45)') == [45.0] and numbers('(4.50)') == [4.5]


@pytest.mark.parametrize(
	'pred,ok',
	[
		('+6 percentage points (from 73% in 2023 to 79% in 2024)', True),
		('6 (73% -> 79%) [rounded]', True),
		('6', True),
		('6 (or 7)', False),
		('6 or 7', False),
		('6, 7', False),
		('maybe 6 (from 73 to 79)', False),
		('7 (6 if rounded down)', False),
		('$13,845,011 increase (6% of burn)', False),
	],
)
def test_a_single_leading_value_with_its_working_in_parentheses(pred, ok):
	assert grade_answer({'answer_type': 'number', 'answer': '6', 'abs_tol': 0.5, 'prompt': 'Change in share, in percentage points?'}, pred) is ok


@pytest.mark.parametrize(
	'task,pred,ok',
	[
		({'answer_type': 'number', 'answer': '4.5'}, '−4.5', False),  # typographic minus is a minus
		({'answer_type': 'number', 'answer': '-4.5'}, '−4.5', True),
		({'answer_type': 'number', 'answer': '-4.5'}, '(4.5)', True),  # a bare parenthesised answer is an accounting negative
		({'answer_type': 'number', 'answer': '1200', 'prompt': 'Revenue, in millions?'}, '1.2 billion', True),
		({'answer_type': 'number', 'answer': '1200', 'prompt': 'Revenue, in millions?'}, '$1.2B', True),
		({'answer_type': 'number', 'answer': '1200', 'prompt': 'Revenue, in millions?'}, '40 mln', False),
		({'answer_type': 'contains', 'answer': 'Paris'}, 'Paris is probably the answer, but it could also be Lyon', False),
		({'answer_type': 'contains', 'answer': 'Paris'}, 'Is it Paris? I think Lyon', False),
		({'answer_type': 'contains', 'answer': 'Paris'}, 'The capital is Paris', True),
		({'answer_type': 'exact', 'answer': 'A'}, 'A', True),  # an option letter is not an article
		({'answer_type': 'exact', 'answer': 'A'}, 'B', False),
		({'answer_type': 'set', 'answer': 'A, C'}, 'A, C', True),
		({'answer_type': 'date', 'answer': '2024-03-13'}, '13/03/2024', True),
		({'answer_type': 'date', 'answer': '2024-03-13'}, '13th of March 2024', True),
	],
)
def test_reviewer_found_grading_cases(task, pred, ok):
	assert grade_answer(task, pred) is ok


@pytest.mark.parametrize('pred,score', [('0.7 (70%)', 0.91), ('70% (0.7)', 0.91), ('7/10', 0.91), ('0.35 probability in 2025', 0.0), ('8/5', 0.0)])
def test_prob_restated_once_is_one_probability(pred, score):
	assert grade_answer({'answer_type': 'prob', 'answer': 'yes'}, pred) == pytest.approx(score)


def test_unreadable_golds_raise():
	for t in ({'answer_type': 'prob', 'answer': 'Resolved yes'}, {'answer_type': 'date', 'answer': 'March 2024'}):
		with pytest.raises(ValueError):
			grade_answer(t, 'yes')


@pytest.mark.parametrize(
	'task,pred,ok',
	[
		({'answer_type': 'exact', 'answer': 'Vitamin A'}, 'Vitamin', False),  # a trailing letter is not an article
		({'answer_type': 'exact', 'answer': 'the Fed'}, 'Fed', True),
		({'answer_type': 'contains', 'answer': 'Hepatitis A'}, 'Hepatitis', False),
		({'answer_type': 'number', 'answer': '5'}, '5 (maybe 6)', False),  # a hedge in brackets is still a hedge
		({'answer_type': 'number', 'answer': '1200', 'prompt': 'Net income in USD millions?'}, '1.2 billion', True),
	],
)
def test_second_review_grading_cases(task, pred, ok):
	assert grade_answer(task, pred) is ok


@pytest.mark.parametrize('pred', ['1.5', 'not 60%', 'No, 0.8'])
def test_prob_rejects_unreadable_numbers(pred):
	assert grade_answer({'answer_type': 'prob', 'answer': 'yes'}, pred) == 0.0


@pytest.mark.parametrize('pred, score', [('1' * 4301 + '/' + '1' * 4301, 1.0), ('1/' + '1' * 4301, 0.0), ('1' * 4301 + '/1', 0.0)])
def test_prob_overlong_fraction_does_not_raise(pred, score):
	assert grade_answer({'answer_type': 'prob', 'answer': 'yes'}, pred) == score
