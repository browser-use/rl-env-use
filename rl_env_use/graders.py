"""Programmatic graders. Deterministic, free, and the primary reward for answer-type tasks.
Every grader fails closed on hedging: an answer offering several candidates never earns reward."""

import re
import unicodedata
from datetime import date, datetime
from urllib.parse import unquote_plus

_ARTICLES = re.compile(r'^(a|an|the)\s+')  # a leading article only: 'Vitamin A' is not 'Vitamin'
_NUM = re.compile(r'[-+]?\d[\d,_]*(?:\.\d+)?|[-+]?\.\d+')
_SCALE = {'thousand': 1e3, 'million': 1e6, 'mio': 1e6, 'mn': 1e6, 'mln': 1e6, 'billion': 1e9, 'bn': 1e9, 'bln': 1e9, 'trillion': 1e12}
_SUFFIX = {'k': 1e3, 'm': 1e6, 'b': 1e9}  # only when attached: "1.2M", never "5 m" (metres)
_DATE_FMTS = (
	'%Y-%m-%d',
	'%d %B %Y',
	'%B %d, %Y',
	'%B %d %Y',
	'%d %b %Y',
	'%b %d, %Y',
	'%b %d %Y',
	'%d.%m.%Y',
	'%m/%d/%Y',
	'%d/%m/%Y',
	'%Y/%m/%d',
)  # slashes read US-first; 13/03/2024 can only be day-first
_MONTHS = '(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'
_DATE_RE = re.compile(
	rf'\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}} {_MONTHS},? \d{{4}}|{_MONTHS} \d{{1,2}},? \d{{4}}|\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{4}}/\d{{2}}/\d{{2}}', re.I
)
_NEG_WORDS = r'not|no|never|neither|nor|cannot|can t|isn t|aren t|wasn t|don t|doesn t|didn t|won t|shouldn t|mustn t|avoid|without|instead of|rather than|deprecated|unsupported'
_NEG = re.compile(rf'\b({_NEG_WORDS})\s+(\S+\s+){{0,3}}$')  # negation within 4 words before the claim


def norm(s) -> str:
	"""Casefold, strip accents (keeping non-Latin scripts), drop punctuation except decimal points, drop articles."""
	s = ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).casefold()
	s = re.sub(r'[^\w\s.]', ' ', s).replace('_', ' ')
	s = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', s)
	s = ' '.join(s.split())
	return _ARTICLES.sub('', s) or s  # 'A' as an option letter is an answer, not an article


_CUR = r'[$€£]'
_AMT = r'\d[\d,]*(?:\.\d+)?'


def signed(text: str) -> str:
	"""Negatives as finance writes them, as plain signed numbers: -$4.50, – $4.50, ($4.50), $(4.50) -> $-4.50."""
	text = re.sub(rf'(?<![\w.])[-\u2212\u2013]\s?({_CUR})\s?(?={_AMT})', r'\1-', text)
	return re.sub(rf'\(\s?({_CUR})\s?({_AMT})\s?\)|({_CUR})\s?\(\s?({_AMT})\s?\)', lambda m: f'{m.group(1) or m.group(3)}-{m.group(2) or m.group(4)}', text)


def numbers(s, scale: bool = True) -> list[float]:
	text = re.sub(r'(?<![\w.])[\u2212\u2013](?=\d)', '-', str(s).lower())  # a typographic minus is a minus; 10\u201312 stays a range
	out, text = [], signed(text)
	for m in _NUM.finditer(text):
		pre, g = text[max(0, m.start() - 2) : m.start()], m.group()
		if pre[-1:].isalpha():
			if g[0] in '+-':
				g = g[1:]  # 'top-5', 'GPT-4': the hyphen separates, it is not a sign
			elif not re.fullmatch(r'\W?v', pre):
				continue  # digits inside identifiers are not quantities (SCN5A, TP53, rs2017481); a version prefix is fine (v2.1)
		try:
			v = float(g.replace(',', '').replace('_', ''))
		except ValueError:
			continue
		word = re.match(r'\s*([a-z]+)', text[m.end() :])
		attached = re.match(r'([kmb])(?![a-z])', text[m.end() :])
		if not scale:
			pass
		elif word and word.group(1) in _SCALE:
			v *= _SCALE[word.group(1)]
		elif attached:
			v *= _SUFFIX[attached.group(1)]
		out.append(v)
	return out


def _to_date(c: str) -> date | None:
	for f in _DATE_FMTS:
		try:
			return datetime.strptime(re.sub(r'(\d) ([A-Za-z]+),', r'\1 \2', c.strip()).replace('Sept ', 'Sep '), f).date()
		except ValueError:
			pass
	return None


def dates(s) -> list[date]:
	"""All distinct dates mentioned, in order."""
	s = re.sub(r'(\d)\s+of\s+', r'\1 ', re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', str(s).strip()), flags=re.I)  # 5th of March
	whole = _to_date(s)
	found = [whole] if whole else [d for d in map(_to_date, _DATE_RE.findall(s)) if d]
	return list(dict.fromkeys(found))


def parse_date(s) -> date | None:
	d = dates(s)
	return d[0] if d else None


def _items(s, prompt: str = '') -> set[str]:
	if isinstance(s, (list, tuple, set)):
		return {norm(x) for x in s if norm(x)}
	out = set()
	s = re.sub(r'https?://\S+', ' ', str(s))  # URLs are citations, not list items
	for x in re.split(r'[;,/\n]|\band\b|\bor\b|&', s):
		m = re.match(r'^([^:]{1,40}):\s*(\S.*)$', x.strip())
		# "WebVoyager: Apache-2.0" -> "Apache-2.0", but only when the label names an entity the question mentions
		out.add(norm(m.group(2)) if m and norm(m.group(1)) and f' {norm(m.group(1))} ' in f' {norm(prompt)} ' else norm(x))
	return out - {''}


def exact(pred, gold, aliases=()) -> bool:
	p = norm(pred)
	return bool(p) and any(p == norm(g) for g in [gold, *aliases])


def contains(pred, gold, aliases=()) -> bool:
	"""Gold appears as whole tokens, is never negated, and the answer commits to it: no 'X or Y', no bare candidate lists."""
	p = f' {norm(pred)} '
	golds = [f' {norm(g)} ' for g in [gold, *aliases] if norm(g)]
	if not golds or _COMMIT.search(str(pred)):
		return False
	if not all(any(g in f' {alt} ' for g in golds) for alt in p.split(' or ')):
		return False
	items = _items(pred)
	bare_list = len(items) > 1 and all(len(i.split()) <= 4 for i in items)  # "Paris, Lyon, Berlin" is a hedge
	if bare_list and not all(any(g in f' {i} ' for g in golds) for i in items):
		return False
	return not any(_NEG.search(p[: m.start() + 1]) for g in golds for m in re.finditer(re.escape(g), p))


_UNIT = re.compile(r'\bin (?:(?:[$€£]|usd|us\$|eur|gbp|chf|cad)\s?)?(thousands|millions|billions|1,000s|000s|1,000,000s|000,000s)', re.I)
_UNITS = {'thousands': 1e3, 'millions': 1e6, 'billions': 1e9, '1,000s': 1e3, '000s': 1e3, '1,000,000s': 1e6, '000,000s': 1e6}


_HEDGE = re.compile(r'\b(either|maybe|possibly|alternatively|between)\b', re.I)
_COMMIT = re.compile(
	r'\?|\b(either|maybe|possibly|perhaps|probably|alternatively|could (also )?be|might be|or else|i think|not sure|unclear)\b|\b(but|although|though) .{0,40}\b(also|could|might|possible)\b',
	re.I,
)  # naming candidates in prose is a hedge too


def number(pred, gold, rel_tol=0.0, abs_tol=0.0, unit: str | None = None) -> bool:
	"""`unit`: the scale the prompt asks for ('in thousands' -> 'thousands'); an answer restating it is read in that unit."""
	g = numbers(gold)
	if len(g) != 1:
		raise ValueError(f'number gold must contain exactly one number: {gold!r}')
	pred = re.sub(r'^\s*\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)\s*$', r'-\1', str(pred))  # an answer that is only (4.5) is an accounting negative
	p = numbers(pred)
	head = re.split(r'\s*[(\[]', str(pred), maxsplit=1)[0]
	if len({round(x, 9) for x in p}) > 1 and len(numbers(head)) == 1 and not _HEDGE.search(str(pred)) and not re.search(r'\bor\b', str(pred), re.I):
		p = numbers(head)  # '6 percentage points (from 73% to 79%)': one answer, its working in parentheses
	if len({round(x, 9) for x in p}) != 1:  # hedging with several different numbers fails
		return False
	close = lambda a, b: abs(a - b) <= max(abs_tol, rel_tol * abs(b)) + 1e-9  # noqa: E731
	if close(p[0], g[0]):
		return True
	pu = numbers(pred, scale=False)  # an answer that states its own scale (1.2 billion, $1,200 million) is read in the prompt's unit
	return bool(unit) and numbers(gold, scale=False) == g and len(pu) == len(p) and p[0] != pu[0] and close(p[0] / _UNITS[unit.lower()], g[0])


def date_eq(pred, gold) -> bool:
	p, g = dates(pred), parse_date(gold)
	if g is None:
		raise ValueError(f'date gold must be a full date: {gold!r}')
	return p == [g]


def set_eq(pred, gold, min_recall=1.0, prompt: str = '') -> bool:
	p, g = _items(pred, prompt), _items(gold)
	if not g:
		raise ValueError('empty gold set')
	hit = len(p & g)
	return hit / len(g) >= min_recall and hit == len(p)


def prob_score(pred, gold) -> float:
	"""Forecasts: 1 - Brier. The answer is a probability that the gold event happened (0.7, 70%, or yes/no as 1/0);
	an unparseable, out-of-range or hedged answer scores 0 (Brier 1)."""
	g = norm(gold)
	if g not in ('yes', 'true', '1', 'no', 'false', '0'):
		raise ValueError(f'forecast gold must be yes or no: {gold!r}')
	p = prob_of(pred)
	return 0.0 if p is None else 1 - (p - (1.0 if g in ('yes', 'true', '1') else 0.0)) ** 2


def prob_of(pred) -> float | None:
	if pred is None:
		return None
	w = norm(pred)
	if w in ('yes', 'no', 'true', 'false'):
		return 1.0 if w in ('yes', 'true') else 0.0
	if f := re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*', str(pred)):  # 7/10
		return int(f.group(1)) / int(f.group(2)) if 0 < int(f.group(2)) and int(f.group(1)) <= int(f.group(2)) else None
	text = str(pred).lower()
	if re.search(r'\b(not|no)\b', text) and re.search(r'\d', text):  # 'not 60%', 'No, 0.8': which way is meant is unclear
		return None
	if any(float(m.group(1)) > 1 and not m.group(2) and '.' in m.group(1) for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent)?', text)):
		return None  # 1.5 is neither a probability nor a percent
	ps = [float(m.group(1)) / 100 if m.group(2) or float(m.group(1)) > 1 else float(m.group(1)) for m in re.finditer(r'(?<![\w.])(-?\d+(?:\.\d+)?|-?\.\d+)\s*(%|percent)?', text)]
	if not ps or len({round(x, 6) for x in ps}) != 1:  # '0.7 (70%)' is one probability; '0.35 in 2025' is not
		return None
	return ps[0] if 0 <= ps[0] <= 1 else None


RANGE = re.compile(r'\d[\d,]*\s*[-\u2013]\s*\d[\d,]*')


def grade_answer(task: dict, pred) -> bool | float:
	"""Grade a final answer against a task's gold spec. `None`/empty pred is always a fail. `prob` returns a score in [0, 1]."""
	if pred is None or not str(pred).strip():
		return 0.0 if task['answer_type'] == 'prob' else False
	kind, gold = task['answer_type'], task['answer']
	if kind == 'prob':
		return prob_score(pred, gold)
	if kind == 'exact' and RANGE.fullmatch(gold.strip()):  # '77-149' == 'Start: 77, End: 149' == '77 to 149'
		return re.findall(r'\d+', gold.replace(',', '')) == re.findall(r'\d+', str(pred).replace(',', ''))
	if kind == 'exact':
		return exact(pred, gold, task.get('aliases', ()))
	if kind == 'contains':
		return contains(pred, gold, task.get('aliases', ()))
	if kind == 'number':
		u = _UNIT.search(task.get('prompt', ''))
		return number(pred, gold, task.get('rel_tol', 0.0), task.get('abs_tol', 0.0), u.group(1) if u else None)
	if kind == 'date':
		return date_eq(pred, gold)
	if kind == 'set':
		return set_eq(pred, gold, task.get('min_recall', 1.0), task.get('prompt', ''))
	raise ValueError(f'unknown answer_type {kind!r}')


# Trace-level checks: failure modes found by error analysis that a deterministic check can see (cheaper and more
# reliable than an LLM judge). `grounded` catches answers taken from search snippets or third-party copies.
def page_key(url: str) -> str:
	u = re.sub(r'^https?://(www\.)?', '', str(url).strip().lower()).split('#')[0].split('?')[0].rstrip('/')
	m = re.match(r'arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})', u)
	return f'arxiv:{m.group(1)}' if m else u


def visited_urls(trace: list[dict]) -> list[str]:
	"""Pages the agent actually observed at a step (not tabs it opened but never looked at)."""
	return [m.group(1) for t in trace if t['role'] == 'assistant' for m in [re.match(r'\[step \d+ @ (\S+)\]', t['content'])] if m]


# Actions whose result is page content the agent read. Never `done`/`input`/`write_file`: those echo the policy's
# own text, and counting them would let it ground itself.
READ_ACTIONS = {'extract', 'find_text', 'search_page', 'find_elements', 'dropdown_options', 'scroll'}
# Domains where anyone can publish: a page there only reads a source in the same owner/repo, never someone else's copy.
SHARED_HOSTS = {'github.com', 'gist.github.com', 'api.github.com', 'gitlab.com', 'huggingface.co'}


def _observations(trace: list[dict]) -> list[tuple[str, str, str]]:
	"""(page_key, text, url) for every page-content result, attributed to the page the agent was on at that step."""
	out, url, pending = [], None, []
	for t in trace:
		if t['role'] == 'assistant':
			m = re.match(r'\[step \d+ @ (\S+)\]', t['content'])
			url, pending = (m.group(1) if m else url), []
		elif t['role'] == 'tool_call':
			pending.append(t.get('name'))
		elif t['role'] == 'tool_result':
			name = pending.pop(0) if pending else None
			if url and name in READ_ACTIONS:
				out.append((page_key(url), str(t['content']), url))
	return out


def _space(k: str) -> str:
	"""The owner/repo a page on a shared host belongs to: github.com/o/r/blob/x and api.github.com/repos/o/r -> github.com/o/r,
	huggingface.co/api/datasets/o/n -> huggingface.co/datasets/o/n."""
	p = k.split('/')
	if p[0] == 'api.github.com' and p[1:2] == ['repos']:
		p = ['github.com', *p[2:]]
	if p[0] == 'huggingface.co' and p[1:2] == ['api']:
		p = [p[0], *p[3 if p[2:3] == ['models'] else 2 :]]
	return '/'.join(p[: 4 if p[1:2] in (['datasets'], ['spaces']) else 3])


def grounded(evidence: list[dict], trace: list[dict]) -> bool:
	"""Every fact the answer needs was read at its source. A fact counts as read if the agent observed exactly its
	source page (where the value was verified at scout time), or page content it extracted on that page, a sub-page,
	or a listing or official API on the same site contains the value. On a shared host (github.com, huggingface.co) the same site
	means the same owner/repo. Search snippets, third-party copies and the policy's own outputs never count."""
	seen = {page_key(u) for u in visited_urls(trace)}
	# plain sub-pages (e.g. a dataset's /viewer) of a specific source page; no query strings (issues/new?title=ANSWER),
	# no create/edit/search actions, and never under a bare site root
	views = {page_key(u) for u in visited_urls(trace) if '?' not in u and not re.search(r'/(new|edit|search|compare|fork)(/|$)', u.split('#')[0].lower())}
	obs = _observations(trace)

	def host(k):  # the source's own domain, incl. its official API (rest.uniprot.org reads uniprot.org)
		if k.startswith('arxiv:'):
			return 'arxiv.org'
		lab = k.split('/')[0].split('.')
		return '.'.join(lab[-3:] if len(lab) > 2 and len(lab[-1]) == 2 and lab[-2] in ('co', 'ac', 'gov', 'org', 'com', 'edu') else lab[-2:])

	def same_site(k, s) -> bool:
		h = k.split('/')[0]
		if h in SHARED_HOSTS:
			return _space(k) == _space(s)
		if h.endswith('.github.io'):
			return h == s.split('/')[0]
		return host(k) == host(s)

	def has(v, text) -> bool:
		d = dates(v)
		return f' {norm(v)} ' in f' {norm(text)} ' or (len(d) == 1 and d[0] in dates(text))

	def shown(v, s) -> bool:
		if not v:
			return False
		# a page whose own URL carries the value (a search for the answer) only echoes the policy's query back
		raw = ' '.join(txt for k, txt, u in obs if same_site(k, s) and not has(v, unquote_plus(u.partition('?')[2])))
		return has(v, raw)

	def ok(e):
		s = page_key(e['url'])
		sub = '/' in s and any(v.startswith(s + '/') for v in views)
		return s in seen or sub or shown(e.get('value'), s)

	return bool(evidence) and all(ok(e) for e in {e['url']: e for e in evidence}.values())


def max_repeats(trace: list[dict]) -> int:
	from collections import Counter

	calls = Counter((t.get('name'), t['content']) for t in trace if t['role'] == 'tool_call')
	return max(calls.values(), default=0)
