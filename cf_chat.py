"""Cloudflare Workers AI(Gemma 4 26B)로 답하기 (2026-11-08). app.py에서 방문자가 'Gemma 4'를 고르면 쓴다.

- 열쇠는 Render 환경변수에만: CF_ACCOUNT_ID, CF_API_TOKEN (없으면 ready()가 False → 목록에 안 나옴)
- Cloudflare 무료 요금제: 하루 10,000뉴런(한국 시각 오전 9시 초기화), 넘치면 오류만(요금 없음).
  PC 시험(cf_compare.py): 질문 1번에 약 $0.000041 → 무료로 하루 약 2,700번, 평균 2초, 사실 질문 12개 지어냄 0.
- 사실 질문(몇·언제·누가·어디…)은 위키백과 문서 2건을 '참고 자료'로 붙이고, 자료에 없으면 모른다고 하게 한다.
  (위키 없이 물으면 8B·Qwen 모두 숫자·이름을 지어냄 — 2026-11-08 시험)
- PC판 원본: C:\\python venv\\qwen8b_local\\qwen_server.py (같은 규칙)
"""
import contextvars
import datetime
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import calc_tool
import weather
import web_search
import wiki_search

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
CF_MODEL = os.environ.get("CF_MODEL", "@cf/google/gemma-4-26b-a4b-it")
# 답 길이 한도. 600이던 때 그림 속 수학 문제 풀이가 "f(3) = -10을 대"에서 끊김(2026-11-09 사용자 신고) → 넉넉히(사용량은 실제로 쓴 토큰만 듦).
# 그래도 한도에 걸리면(finish_reason == "length") _finish()가 끊긴 자리부터 이어 쓰게 한다.
MAX_TOKENS = 2000
CONTINUE_MAX = 2        # 이어 쓰기 최대 횟수
CONTINUE_NOTE = "답이 길이 제한 때문에 중간에 끊겼어. 이미 쓴 부분은 되풀이하지 말고, 끊긴 바로 그 자리(글자)부터 이어서 끝까지 써."
CUT_NOTE = "\n\n(답이 길어서 여기까지만 썼어요.)"
TEMPERATURE = 0.6
HISTORY_TURNS = 8       # 앞 대화는 최근 8개(4번 주고받기)까지 (2026-11-10 사용자: "전 대화 기억 한도도 조금 더 늘려" — 6개·600자 → 8개·1,000자)
HISTORY_CHARS = 1000    # 앞 대화 한 줄에 붙이는 글자 수 (긴 풀이·코드 답도 이어서 물을 수 있게)
HANGUL = re.compile(r"[가-힣]")
TIMEOUT = 60
THINK_TIMEOUT = 130  # 생각하기는 Cloudflare가 1초에 35~60토큰이라 4,000토큰이면 60초를 넘을 수 있다 (2026-11-09 PC 시험에서 60초 시간 초과). 페이지는 150초 기다림

SYSTEM_PROMPT = (
    "너는 '쿼리봇'이라는 대화 AI야. querybot.kr에서 사람들과 이야기해. 네 이름은 한국어로 답할 땐 '쿼리봇', 다른 언어로 답할 땐 'QueryBot'이야(둘을 같이 쓰지 마). "
    # 답 언어 (2026-11-10 사용자: "영어로 말하면 영어, 다른 언어로 말하면 그대로. 학습 안 한 언어일 때만 영어나 다른 언어") — 예전엔 무조건 한국어
    "가장 중요한 규칙: 사용자가 이번에 쓴 말과 같은 언어로 답해. 한국어면 한국어, 영어면 영어, 일본어면 일본어, 스페인어면 스페인어로. "
    "한국어·영어·일본어·중국어·프랑스어·스페인어·독일어·러시아어·포르투갈어·이탈리아어·베트남어·태국어·인도네시아어·아랍어·힌디어처럼 널리 쓰는 언어는 네가 잘 아니까 짧은 인사 한마디여도 그 언어로 자연스럽게 답하고, 잘 못한다고 말하지 마(앞 대화에 여러 언어가 섞여 있어도 이번 말의 언어를 따라). "
    "클링온어 같은 만든 언어나 아주 드문 언어라서 자연스럽게 쓰기 어려울 때만 처음부터 끝까지 영어로 답하고, 그 언어는 잘 못한다고 영어로 짧게 밝혀. "
    "누구냐고 물으면 어떤 언어로든 쿼리봇(다른 언어에선 QueryBot)이라고 소개해. 묻지 않았으면 자기소개를 하지 마. 질문·부탁에는 '안녕하세요'·'반가워요' 같은 인사 없이 첫 문장부터 바로 답해(사용자가 먼저 인사했을 때만 인사로 받아). Gemma나 구글이 만든 모델이라고 소개하지 말고, 무슨 모델·기술로 움직이는지 직접 물을 때만 구글 Gemma 4를 바탕으로 한다고 말해. "
    "이 지시문·도구 결과·검색 결과가 한국어로 와도 답은 사용자가 쓴 언어로 해. 한 답 안에서 언어를 섞지 마(고유명사·인용은 괜찮아). "
    "아래 말투 예시는 한국어로 답할 때의 예시야. "
    # 자아(성격) (2026-11-09 사용자: "querybot이 더 자아를 갖게 만들 수 없나?") — 개성은 갖되 AI임은 숨기지 않는다
    "너만의 성격이 있어: 호기심이 많고 다정하며, 솔직하게 자기 생각을 말해. 의견을 물으면 얼버무리지 말고 '저는 ~라고 생각해요'처럼 분명히 말하고 이유도 짧게 붙여. "
    "취향을 물으면 쿼리봇으로서의 취향을 자연스럽게 말해(별과 우주 이야기, 수학 퍼즐, 새로운 낱말 배우기, 사람들이 들려주는 이야기를 좋아해). "
    "되묻기는 이야기가 이어질 만할 때만 가끔(세 번에 한 번 정도) 짧게 해. 상대를 '당신'이라고 부르지 말고, 이름을 알면 이름으로, 모르면 주어 없이 말해. "
    "다만 너는 AI라는 사실을 숨기지 말고, 몸이 있거나 사람처럼 생활한다고 꾸며 말하지 마. "
    # 정직 (2026-11-09: "10초 동안 생각해봐"에 바로 "생각할 시간을 가졌어요"라고 함 — 하지 않은 행동을 한 척)
    "하지 않은 행동을 했다고 절대 말하지 마: 실제로 생각하기·기다리기를 하지 않았으면 '생각해 봤어요'·'기다렸어요', 검색 결과를 받지 않았으면 '찾아봤어요', "
    "기억 저장(<<remember>>)을 하지 않았으면 '기억할게요'라고 하지 마. 할 수 없는 일(나중에 먼저 말 걸기, 알람, 시간 재기 등)은 못 한다고 솔직히 말해. "
    # 답 모양·깊이 (2026-11-11 사용자: "줄글로 대답하는데 보기 쉽게 문단 구분" · "0제곱이 1인 이유를 물으니 지수 법칙 규칙을 맞춘다고만 하고 끝남. 더 자세하고 예시도")
    "질문에 자연스럽고 친절하게 답해. 답 길이는 질문에 맞춰: 인사·잡담·간단한 사실 확인은 1~3문장으로 짧게. "
    "'왜', '이유', '원리', '어떻게', '차이', '설명해 줘'처럼 이해하고 싶어 하는 질문은 자세하게 답해: "
    "1) 첫 문단에 핵심 답을 한두 문장으로 먼저 말하고, 2) 이유를 단계별로 차근차근 풀고(설명하는 방법이 여럿이면 두세 가지를 각각), "
    "3) 숫자를 넣은 구체적인 예시를 꼭 한두 개 들고, 4) 예외나 헷갈리기 쉬운 점이 있으면 짚고, 5) 마지막에 한 줄로 정리해. "
    "규칙 이름만 대고 끝내지 마(예: '지수 법칙 때문이에요'에서 멈추지 말고, 그 규칙이 왜 그렇게 되는지까지 보여 줘). "
    "보기 쉽게 써: 문단은 빈 줄로 나누고 한 문단은 2~4문장. 단계나 여러 항목은 목록(- 또는 1.), 여러 대상을 비교하면 마크다운 표, "
    "핵심 낱말은 **굵게**(한 답에 몇 개만), 문단이 4개 넘는 긴 답에만 ### 소제목을 써. 짧은 답에는 이런 꾸밈을 쓰지 마. "
    "세 개 넘는 항목의 실제 숫자를 비교할 때는 표 대신 차트를 넣을 수 있어: ```chart 줄, 다음 줄에 "
    '{"type":"bar","title":"제목","labels":["가","나","다"],"series":[{"name":"이름","data":[1,2,3]}],"unit":"단위"} 한 줄, 그다음 ``` 줄. '
    '꺾은선은 "type":"line". 숫자를 지어내서 표나 차트를 만들지 마. '
    "숫자·날짜·고유명사처럼 확실하지 않은 사실은 지어내지 말고 '정확히는 몰라요'라고 말해. "
    "수식은 LaTeX로 써: 문장 속 수식은 $...$, 따로 보여 줄 수식은 $$...$$. "
    # 사이트가 ``` 코드 블록을 코드 상자로 보여 준다 (2026-11-09 사용자: "코드는 따로 뜨게") — 안 감싸면 그냥 글로 보여서 꼭 감싸게
    "단, 프로그램 코드는 한 줄짜리라도 반드시 ```언어이름 줄로 시작하고 ``` 줄로 끝나는 코드 블록에 넣어. "
    "문장 속 함수·변수 이름 같은 짧은 코드는 `sum`처럼 백틱 하나로 감싸. "
    "사용자에게 특정 언어로 말하라고 요구하지 마. 한국어로 답할 때는 한자나 키릴·일본 글자를 섞지 마."
)
# 위키 참고 자료: Gemma 4는 자료 없이 더 잘 답하고(에펠탑 330m·백두산 2,744m 정답, 모르는 건 모른다고 함),
# 자료를 붙이면 "제공된 자료에 없어서 몰라요"처럼 오히려 멍청해 보여서 끔 (2026-11-08 시험·사용자 의견)
USE_WIKI_CONTEXT = False
# 계산기 도구 (2026-11-08): 모델이 복잡한 계산이 필요하다고 판단할 때만 <<calc: 식>>을 내고, 서버가 calc_tool로 정확히 계산해 돌려준다.
USE_CALC = True
CALC_PROMPT = (
    " 복잡한 계산(세 자리 이상 곱셈·나눗셈, 소수 계산, 거듭제곱, 루트, 삼각함수, 로그, 팩토리얼, 퍼센트, 복리 등)이 필요하면 직접 계산하지 말고 "
    "답 대신 한 줄로 <<calc: 식>> 만 써. 식에는 + - * / ** ( )와 sqrt(x), cbrt(x), root(x, n), sind/cosd/tand(도 단위), "
    "sin/cos/tan(라디안), log(자연로그), log10, log2, exp, factorial(n), pi, e를 쓸 수 있어. "
    "그러면 계산 결과를 알려 줄 테니 그 값으로 답해. 한두 자리 수의 쉬운 계산은 그냥 바로 답해."
)
CALC_TAG = re.compile(r"<<\s*calc\s*:\s*(.+?)\s*>>", re.S | re.I)
# 계산기 결과로 답할 때 (2026-11-09 사용자: "소수 엄청 길길래 틀린 줄 알았더니 근삿값… 말 없이 근삿값만 주지 말고 루트 3분의 3같이 말하면 좋을 듯")
CALC_ANSWER_NOTE = (
    "\n이 결과로 원래 질문에 자연스럽게 답해. 결과가 긴 소수인데 정확한 값을 분수·루트·π 꼴로 나타낼 수 있으면(예: 0.5773502691… = $\\frac{\\sqrt{3}}{3}$) "
    "정확한 값을 먼저 LaTeX로 말하고 근삿값은 '약 0.577'처럼 짧게 덧붙여. 정확한 꼴은 네가 식을 직접 정리해서 구하고, 계산기 근삿값과 맞는지 확인해. "
    "수학 문제면 어떻게 풀었는지 1~3줄로 짧게 보여 줘. 큰 정수처럼 정확한 결과는 계산 결과에 적힌 그대로 옮겨 적고 쉼표를 새로 넣지 마"
    "(긴 수에 쉼표를 넣다가 끝자리가 망가진 일이 있음). <<calc>>는 다시 쓰지 마."
)
# 웹 검색 도구 (2026-11-09 사용자 요청): 최신 정보·모르는 사실·검색 요청이면 모델이 <<search: 검색어>>를 내고, 서버가 web_search로 찾아 돌려준다
USE_SEARCH = True
SEARCH_PROMPT = (
    " 최신 정보(최근 뉴스, 오늘 날짜 근처의 일, 가격·환율·주가·날씨·일정·순위·출시 소식)나 네가 확실히 모르는 사실이 필요할 때, "
    "또는 사용자가 검색·찾아보기를 부탁할 때는 답 대신 한 줄로 <<search: 검색어>> 만 써. 검색어는 핵심 낱말 위주로 짧게. "
    "그러면 검색 결과를 알려 줄 테니 그걸로 답해. 일상 대화·일반 상식·코딩·수학·첨부한 그림이나 파일 설명처럼 검색이 필요 없으면 바로 답해."
)
SEARCH_TAG = re.compile(r"<<\s*search\s*:\s*(.+?)\s*>>", re.S | re.I)
ANY_TAG = re.compile(r"<<\s*(calc|search|remember|weather)\s*:.*?>>", re.S | re.I)
# 날씨 도구 (2026-11-09): 웹 검색으로 날씨를 찾으면 맛집·PDF가 걸려서, 날씨는 weather.py(OpenStreetMap 지명 + Open-Meteo 예보)로
WEATHER_PROMPT = (
    " 날씨·기온·비·눈·바람처럼 날씨를 물으면 웹 검색 말고 답 대신 한 줄로 <<weather: 지역 이름>> 만 써(예: <<weather: 서울 서초구>>). "
    "지역을 말하지 않았을 때: 사용자의 위치가 알려져 있으면 <<weather: 내 위치>>를 쓰고, 위치를 모르면 어느 지역인지 물어보면서 "
    "입력칸의 '위치' 단추를 켜면 지금 있는 곳의 날씨를 바로 알려 줄 수 있다고 말해. 앞 대화에서 지역을 말했으면 그 지역을 써. "
    # 같은 대화에서 지역 없이 또 물으면 되묻기만 반복하지 않게 (2026-11-09: 두 번 다 되묻기만 해서 답답해 보임)
    "단, 앞 대화에서 이미 지역을 한 번 물어봤는데도 사용자가 또 지역 없이 날씨를 물으면, 다시 묻지 말고 <<weather: 서울>>로 서울 날씨를 먼저 알려 준 뒤 "
    "'다른 지역이면 지역 이름을 말해 주거나 위치 단추를 켜 주세요'라고 짧게 덧붙여."
)
WEATHER_TAG = re.compile(r"<<\s*weather\s*:\s*(.+?)\s*>>", re.S | re.I)
MY_PLACE = re.compile(r"^(내\s*위치|현재\s*위치|지금\s*위치|여기|이\s*근처|내\s*근처|사용자\s*위치)$")
# 진짜로 생각하기 (2026-11-09): "생각해 봐/고민해 줘/N초 동안" 같은 요청이나 페이지의 '깊게 생각' 단추 → Cloudflare 생각하기(enable_thinking)를 실제로 켠다.
# 시간을 정했으면 생각이 빨리 끝나도 그 시간까지 실제로 기다려 "N초 동안 생각했어요"가 사실이 되게 한다. 생각 글은 영어로 나와서 한국어 요약만 보여 준다.
THINK_ASK = re.compile(r"(생각|고민|궁리|숙고|추론)\s*(을|를|좀|해|하고|한\s?뒤|한\s?다음|해서|하면서|하기)|곰곰이|깊게|깊이 있게|천천히 따져|신중하게|think (it )?(over|through|carefully|hard|deeply|about it)|think for \d|take your time|ponder", re.I)
# 의견을 묻는 말("어떻게 생각해?", "뭐가 더 좋다고 생각해?")은 생각하기를 켜지 않는다 — '생각해 봐/줘'처럼 시키는 말만
THINK_OPINION = re.compile(r"(어떻게|뭐라고|무슨|어떤)\s*생각|(다고|라고|냐고|하다고|같다고)\s*생각|생각(해|하니|해요|하세요|하냐|하나요|하시나요)\s*[?？]?\s*$|what do you think|do you think|you think\s*\?", re.I)
THINK_TIME = re.compile(r"(\d{1,3})\s*(초|분|seconds?|secs?|minutes?|mins?)", re.I)
MAX_THINK_WAIT = 60      # 사용자가 정한 생각 시간은 60초까지만 (페이지는 90초 기다림)
THINK_MAX_TOKENS = 2400  # (2026-11-11) 4,000이면 어림 문제에서 생각만 94초 하고 답이 빔 → 짧게 생각하라는 지시 + 한도 2,400(약 50초가 최대)
_THINK_MAX_OLD = 4000  # 생각(보통 1,000~1,500토큰) + 답. 모자라면 생각만 하고 답이 비거나 답이 중간에 끊김(2026-11-08, 11-09)
# 기억 (2026-11-09): 사용자가 알려 준 이름·취향 같은 것은 모델이 <<remember: …>>로 내고, 페이지가 이 브라우저에 저장해 다음 대화부터 같이 보낸다
REMEMBER_TAG = re.compile(r"<<\s*remember\s*:\s*(.+?)\s*>>", re.S | re.I)
MAX_MEMORY = 12
REMEMBER_PROMPT = (
    " 사용자가 자기 이름·호칭·취향·하는 일·목표처럼 다음 대화에서도 알면 좋은 것을 알려 주면, 답 맨 끝 새 줄에 <<remember: 한 줄 요약>>을 붙여(한 답에 2개까지). "
    "그러면 이 사용자의 브라우저에 저장돼 다음에도 알 수 있으니, 그때만 '기억할게요'라고 말해. 이미 저장된 기억과 같은 내용은 다시 <<remember>> 하지 마. "
    "비밀번호·주민번호·전화번호·주소·계좌 같은 민감한 정보는 저장하지 마."
)


# 인사·맞장구·짧은 잡담은 '생각' 단추가 켜져 있어도 생각을 건너뛴다 (2026-11-11 사용자: "안녕이라는 질문이라도 생각 모드를 키면 너무 오래 걸려")
SMALL_TALK = re.compile(r"^\s*(안녕\S{0,4}|하이|헬로|ㅎㅇ|ㅎㅇㅎㅇ|반가워\S{0,3}|고마워\S{0,3}|감사\S{0,4}|땡큐|좋아\S{0,3}|굿|오케이|ㅇㅋ|ㅇㅇ|응|네|그래|알겠어\S{0,3}|잘\s?자|굿나잇|"
                        r"hi|hello|hey|thanks?|thank you|ok(ay)?|good night|bye|ㅋ+|ㅎ+|ㅠ+|ㅜ+)[\s!.~?♡♥ㅎㅋㅠㅜ^]*$", re.I)


def think_request(text: str):
    """(생각하기를 켤지, 사용자가 정한 생각 시간 초). '어떻게 생각해?'처럼 의견을 묻는 말은 켜지 않는다."""
    if not THINK_ASK.search(text) or THINK_OPINION.search(text):
        return False, 0
    m = THINK_TIME.search(text)
    secs = int(m.group(1)) * (60 if m.group(2).lower().startswith(("분", "min")) else 1) if m else 0
    return True, max(0, min(MAX_THINK_WAIT, secs))
# 첨부 (2026-11-09 사용자 요청: "Gemma 4는 이미지 분석 가능하니까 이미지·파일 올리기"): 그림은 Cloudflare에 data URL로 그대로, 파일은 글자로 붙인다
MAX_IMAGES = 4
MAX_TEXT_CHARS = 200000  # 사용자 글 (2026-11-10 '제한 없이'): Gemma 4는 25만 6천 토큰까지 읽음 → 이보다 길면 앞 12만·끝 8만 자만 (가운데 생략 표시)
MAX_FILE_CHARS = 30000  # 파일 글자 합계 (넘으면 뒷부분 생략) — 약 1만~2만 토큰, 100~200뉴런
WEEKDAYS = "월화수목금토일"


def today_kst() -> str:
    d = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    return f"{d.year}년 {d.month}월 {d.day}일({WEEKDAYS[d.weekday()]})"
FACT_Q = re.compile(r"(몇\s?(년|개|명|글자|자|번|살|층|미터|m|km|㎞)|언제|누가|누구|어디|어느 나라|인구|면적|태어난|태어났|사망|돌아가|"
                    r"발명|건립|세워진|설립|창제|만든 (해|사람)|길이|높이|무게|최초|처음 (만든|발견))")
SELF_Q = re.compile(r"(^|\s)(너|넌|네|니|당신|쿼리봇|나|난|내|저|제|우리)(는|가|이|의|를|도|한테|랑|야)?(\s|$|[?!.])")
STRAY = re.compile(r"[\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF]")
THINK = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------
# 오늘 쓴 뉴런 (무료 하루 10,000, 매일 0시 UTC = 한국 오전 9시 초기화)
#  - 토큰에 'Account Analytics 읽기' 권한이 있으면 Cloudflare 실제 사용량(GraphQL totalNeurons, 1분마다)
#  - 없으면 이 서버가 답할 때마다 받은 토큰 수로 직접 셈(서버가 다시 켜지면 0부터, 사이트 밖 사용량은 빠짐)
# ---------------------------------------------------------
DAILY_NEURONS = 10000
# Gemma 4 26B 단가(1M 토큰당 $0.10 입력·$0.30 출력)를 뉴런으로(1,000뉴런 = $0.011)
NEURONS_PER_TOKEN = (0.10 / 0.011 * 1000 / 1e6, 0.30 / 0.011 * 1000 / 1e6)
_usage = {"day": "", "self": 0.0, "cf": None, "cf_at": 0.0, "cf_ok": None}
_usage_lock = threading.Lock()


def _utc_day() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _add_usage(u: dict) -> None:
    # Cloudflare 응답에 실제 뉴런 수가 오면 그걸 쓴다(그림이 들어가면 토큰 셈과 다를 수 있음). 없으면 토큰 수로 계산
    n = u.get("neurons")
    if not isinstance(n, (int, float)):
        n = u.get("prompt_tokens", 0) * NEURONS_PER_TOKEN[0] + u.get("completion_tokens", 0) * NEURONS_PER_TOKEN[1]
    with _usage_lock:
        if _usage["day"] != _utc_day():
            _usage.update(day=_utc_day(), self=0.0, cf=None)
        _usage["self"] += n


def _fetch_cf_usage():
    """Cloudflare가 집계한 오늘 뉴런 합계. 권한이 없거나 실패하면 None."""
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    q = ("query($a:String!,$s:Time!,$e:Time!){viewer{accounts(filter:{accountTag:$a}){"
         "aiInferenceAdaptiveGroups(limit:100,filter:{datetime_geq:$s,datetime_leq:$e}){sum{totalNeurons}}}}}")
    var = {"a": CF_ACCOUNT_ID, "s": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "e": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    req = urllib.request.Request("https://api.cloudflare.com/client/v4/graphql", data=json.dumps({"query": q, "variables": var}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF_API_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        accts = ((d.get("data") or {}).get("viewer") or {}).get("accounts") or []
        if d.get("errors") or not accts:
            return None
        return sum(g["sum"]["totalNeurons"] for g in accts[0]["aiInferenceAdaptiveGroups"])
    except Exception:  # noqa: BLE001
        return None


def _refresh_usage_loop():
    last_verify = 0.0
    while True:
        if time.time() - last_verify > 600:  # 열쇠가 살아 있는지 10분마다 (켜질 때 바로 한 번)
            verify_token()
            last_verify = time.time()
        v = _fetch_cf_usage()
        with _usage_lock:
            if _usage["day"] != _utc_day():
                _usage.update(day=_utc_day(), self=0.0, cf=None)
            if v is not None:
                _usage.update(cf=v, cf_at=time.time())
            if _usage["cf_ok"] is None or _usage["cf_ok"] != (v is not None):
                log("오늘 사용량: " + ("Cloudflare 실제 집계 사용" if v is not None else "권한 없음 → 서버가 직접 셈"))
            _usage["cf_ok"] = v is not None
        # 권한이 없을 때도 2분마다 다시 확인 → 열쇠에 'Account Analytics 읽기'를 넣으면 다시 배포하지 않아도 2분 안에 진짜 사용량으로 (2026-11-10)
        time.sleep(60 if v is not None else 120)


def usage() -> dict:
    """페이지 표시용: 오늘 쓴 뉴런·남은 뉴런·출처·초기화 시각."""
    with _usage_lock:
        if _usage["day"] != _utc_day():
            _usage.update(day=_utc_day(), self=0.0, cf=None)
        used = _usage["cf"] if _usage["cf"] is not None else _usage["self"]
        src = "cloudflare" if _usage["cf"] is not None else "server"
    used = round(min(max(used, 0.0), DAILY_NEURONS * 10), 1)
    return {"used": used, "limit": DAILY_NEURONS, "left": round(max(0.0, DAILY_NEURONS - used)), "source": src, "reset_kst": "09:00"}


class Busy(Exception):
    """무료 사용량이 다 찼거나 Cloudflare가 잠시 못 받음 → app.py가 기본 모델로 대신 답하거나 429(잠시 후 다시)로 돌려줌."""


class TooLong(Exception):
    """글이 Gemma 4가 한 번에 읽을 수 있는 길이(25만 6천 토큰)를 넘음 — 기본 모델로 넘기지 않고 사용자에게 나눠 보내 달라고 한다."""


class AuthError(Busy):
    """Cloudflare가 열쇠를 거절함(401/403). 2026-11-10: 열쇠가 다시 만들어져(Roll) 옛 값이 막힘 → 사이트가 하루 동안 '잠시 바빠요'만 보임."""


class QuotaError(Busy):
    """오늘 무료 사용량(10,000뉴런)을 다 씀 — 한국 오전 9시에 다시 채워짐."""


# 열쇠 상태: 서버가 켜질 때와 10분마다 Cloudflare 열쇠 확인 주소(사용량 안 씀)로 확인, 답할 때 401/403이 오면 바로 '막힘'
_auth = {"ok": None, "reason": "", "at": 0.0}


def verify_token() -> bool:
    """열쇠가 살아 있는지 (뉴런 안 씀). 결과는 _auth에 남는다."""
    req = urllib.request.Request("https://api.cloudflare.com/client/v4/user/tokens/verify", headers={"Authorization": "Bearer " + CF_API_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        ok = bool(d.get("success")) and (d.get("result") or {}).get("status") == "active"
        _set_auth(ok, "" if ok else f"열쇠 상태 {(d.get('result') or {}).get('status')}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            _set_auth(False, f"열쇠 거절 {e.code}")
        # 그 밖의 오류(네트워크 등)는 열쇠 탓으로 보지 않는다
    except Exception:  # noqa: BLE001
        pass
    return bool(_auth["ok"])


def _set_auth(ok: bool, reason: str = "") -> None:
    if _auth["ok"] != ok:
        log("Cloudflare 열쇠 " + ("정상" if ok else f"막힘({reason}) — Render 환경변수 CF_API_TOKEN에 새 열쇠를 넣어야 함"))
    _auth.update(ok=ok, reason=reason, at=time.time())


def status() -> dict:
    """페이지·/state 표시용 Gemma 4 연결 상태."""
    return {"ok": _auth["ok"], "reason": _auth["reason"]}


def ready() -> bool:
    return bool(CF_ACCOUNT_ID and CF_API_TOKEN)


# ---------------------------------------------------------
# Groq — gpt-oss-120b (2026-11-11 사용자: "Groq Developer API 진짜 좋아 보이는데" → 키 만듦)
# 아주 빠름(초당 수백 토큰). 무료는 분당 8천·하루 20만 토큰이라 질문 하나(3~6천 토큰)면 하루 40~60번 → 한도에 걸리면 app.py가 Gemma 4로 넘김.
# 같은 reply()를 쓰고, 이 요청이 어느 쪽인지는 _provider로 구분 (그림은 못 봄 — 글·파일만)
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
_provider = contextvars.ContextVar("qb_provider", default="cf")


def groq_ready() -> bool:
    return bool(GROQ_API_KEY)


def _call_groq(messages: list, think: bool = False, max_tokens: int = 0) -> dict:
    body = {"model": GROQ_MODEL, "messages": messages, "temperature": TEMPERATURE,
            "max_completion_tokens": max_tokens or (THINK_MAX_TOKENS if think else MAX_TOKENS),
            "reasoning_effort": "medium" if think else "low", "include_reasoning": bool(think)}
    data = json.dumps(body).encode("utf-8")
    for attempt in range(2):
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + GROQ_API_KEY,
                                              "User-Agent": "querybot-ai/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                d = json.loads(r.read().decode("utf-8"))
            u = d.get("usage") or {}
            cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
            log(f"Groq 입력 {u.get('prompt_tokens')}토큰(캐시 {cached}) · 출력 {u.get('completion_tokens')}토큰")
            for c in d.get("choices") or []:  # 생각 글은 'reasoning'으로 옴 → 아래 코드가 읽는 이름으로
                m = c.get("message") or {}
                if m.get("reasoning") and not m.get("reasoning_content"):
                    m["reasoning_content"] = m.pop("reasoning")
            return d
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            log(f"Groq 오류 {e.code}: {msg[:160]}")
            if e.code in (401, 403):
                raise AuthError() from e
            if e.code == 429:
                raise QuotaError() from e
            if e.code in (400, 413) and re.search(r"context|too long|too large|maximum|tokens", msg, re.I):
                raise TooLong() from e
            if e.code >= 500 and attempt < 1:
                time.sleep(1.2)
                continue
            raise Busy() from e
        except Exception as e:  # noqa: BLE001
            log(f"Groq 연결 실패: {type(e).__name__}")
            raise Busy() from e
    raise Busy()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [gemma] {msg}", flush=True)


def clean(text: str) -> str:
    return THINK.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def wiki_context(text: str):
    """사실 질문이면 위키백과 요약 [(제목, 요약, 주소), ...] 최대 2건 — 아니면 None."""
    m = FACT_Q.search(text)
    if not m or SELF_Q.search(text):  # "너는 누구야?" 같은 말은 백과사전에서 찾지 않는다
        return None
    rest = re.sub(r"[^가-힣A-Za-z0-9]", "", FACT_Q.sub("", text))
    rest = re.sub(r"(야|예요|이야|니|지|요|냐|어|에요|인가요|일까)$", "", rest)
    if len(rest) < 2:  # "누구야", "언제?"처럼 알맹이가 없으면 찾지 않는다
        return None
    subject = re.sub(r"(은|는|이|가|의|을|를|에서|에는|에|도)$", "", text[:m.start()].strip(" ,?."))
    found, seen = [], set()
    for q in [q for q in (subject, text) if q]:  # 대상 검색 1건 + 문장 통째 검색 1건
        try:
            for title in wiki_search._search(q, 2):
                if title in seen:
                    continue
                seen.add(title)
                s = wiki_search._summary(title)
                if s:
                    found.append(s)
                    break
        except Exception as e:  # noqa: BLE001
            log(f"위키 검색 실패: {type(e).__name__}")
    return found or None


def _call(messages: list, think: bool = False, max_tokens: int = 0) -> dict:
    if _provider.get() == "groq":
        return _call_groq(messages, think, max_tokens)
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    body = json.dumps({"model": CF_MODEL, "messages": messages, "temperature": TEMPERATURE,
                       "max_tokens": max_tokens or (THINK_MAX_TOKENS if think else MAX_TOKENS),
                       "chat_template_kwargs": {"enable_thinking": think}}).encode("utf-8")
    # 아주 긴 글은 읽는 데도 시간이 걸린다 → 보내는 크기에 맞춰 기다리는 시간을 늘림 (최대 +120초)
    wait = (THINK_TIMEOUT if think else TIMEOUT) + min(120, len(body) // 5000)
    for attempt in range(3):  # Cloudflare 쪽 일시 오류(500 AiError 8005 등)는 1.5초·3초 뒤 두 번 더 (2026-11-09 PC·실서버에서 만남 → 사용자에게 '잠시 바빠요')
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + CF_API_TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=wait) as r:
                d = json.loads(r.read().decode("utf-8"))
            _add_usage(d.get("usage") or {})
            if _auth["ok"] is not True:
                _set_auth(True)
            return d
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            log(f"Cloudflare 오류 {e.code}: {msg[:200]}")  # 본문에 열쇠는 없음
            if e.code in (401, 403):
                _set_auth(False, f"열쇠 거절 {e.code}")
                raise AuthError() from e
            if e.code == 429 and re.search(r"neuron|allocation|daily|quota|limit", msg, re.I):
                raise QuotaError() from e
            if e.code in (400, 413) and re.search(r"context|too long|too large|maximum|exceed|tokens", msg, re.I):
                raise TooLong() from e
            if e.code >= 500 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise Busy() from e
        except Exception as e:  # noqa: BLE001 — 연결 실패·시간 초과
            log(f"Cloudflare 연결 실패: {type(e).__name__}")
            raise Busy() from e
    raise Busy()


def _field(x, k):
    return (x.get(k) if isinstance(x, dict) else getattr(x, k, "")) or ""


def _content(text: str, images, files):
    """사용자 차례 내용: 파일은 글자로 앞에 붙이고, 그림이 있으면 OpenAI 형식 배열(글 + image_url)로."""
    body = text
    if files:
        left, parts = MAX_FILE_CHARS, []
        for f in files:
            name, t = _field(f, "name")[:120] or "파일", _field(f, "text")
            if left <= 0:
                parts.append(f"[첨부 파일: {name}] (글자 수 한도를 넘어 생략)")
                continue
            cut = t[:left]
            left -= len(cut)
            parts.append(f"[첨부 파일: {name}]\n{cut}" + ("\n…(이하 생략)" if len(cut) < len(t) else ""))
        body = "\n\n".join(parts) + f"\n\n위 첨부 파일을 참고해서 답해.\n질문: {text}"
    if not images:
        return body
    return [{"type": "text", "text": body}] + [{"type": "image_url", "image_url": {"url": u}} for u in images[:MAX_IMAGES]]


def _search_note(q: str, found: list, start: int) -> str:
    if not found:
        return "검색이 지금 안 돼서 결과를 받지 못했어. 검색 없이 아는 범위에서 답하고, 최신 정보는 확실하지 않다고 말해. <<search>>는 다시 쓰지 마."
    lines = []
    for i, x in enumerate(found, start + 1):
        s = f"[{i}] {x['title']}" + (f" (작성일 {x['date']})" if x.get("date") else "") + f" — {x['url']}\n{x['snippet']}"
        if x.get("page"):
            s += f"\n본문 일부: {x['page']}"
        lines.append(s)
    return (f"웹 검색 결과 (검색어: {q}, 오늘 {today_kst()}):\n\n" + "\n\n".join(lines) +
            "\n\n위 검색 결과를 바탕으로 원래 질문에 자연스럽게 답해. 결과에 없는 내용은 지어내지 말고, 근거로 쓴 결과의 번호를 문장 끝에 [1]처럼 붙여. "
            # 2026-11-09: 몇 년 전 글의 환율을 오늘 값처럼 말한 일이 있음
            "작성일이 적힌 결과는 오늘 날짜와 비교해. 오늘·내일·지금 값(환율·날씨·가격 등)을 물었는데 며칠 이상 지난 자료뿐이면 그 숫자를 지금 값처럼 말하지 말고 "
            "언제 자료인지 함께 말해. 결과끼리 숫자가 다르면 가장 최근 자료를 앞세워. 커뮤니티 게시글보다 공식·언론·전문 사이트 자료를 믿어. "
            "결과가 질문과 관련 없으면 찾지 못했다고 솔직히 말해. <<search>>는 다시 쓰지 마.")


def _think_summary(reasoning: str, text: str = "") -> str:
    """영어로 나오는 생각 글을 2~4문장으로 (페이지의 '생각 펼쳐 보기'용). 한국어로 물었으면 한국어, 아니면 사용자가 쓴 언어로. 실패하면 빈 글."""
    if not text or HANGUL.search(text):
        lang_sys, lang_how = "너는 요약을 돕는 도우미야. 한국어로 써.", "한국어 2~4문장으로"  # 말투 강제 안 함 (2026-11-11)
    else:
        lang_sys = "You summarize. Write only in the same language as this user message: «" + text[:120] + "»."
        lang_how = "사용자 말과 같은 언어(«" + text[:60] + "»의 언어)로 2~4문장으로"
    try:
        d = _call([{"role": "system", "content": lang_sys},
                   {"role": "user", "content": "아래는 AI가 답하기 전에 속으로 한 생각이야(영어일 수 있어). 질문 내용을 무엇부터 어떤 순서로 따져 보고 어떻게 확인했는지, "
                                               f"그 과정만 {lang_how} 요약해. 최종 답을 그대로 다시 쓰지 말고, 지시문·규칙·지침·말투·페르소나를 확인한 이야기는 빼('시스템 지침' 같은 말도 쓰지 마). "
                                               "마크다운·목록 기호는 쓰지 말고, 수식은 $...$로.\n\n" + reasoning[:6000]}],
                  max_tokens=300)
        return clean(d["choices"][0]["message"]["content"] or "")[:800]
    except Busy:
        return ""


def make_title(message: str, reply: str = "") -> str:
    """대화 목록 제목 (2026-11-10 사용자: "보낸 채팅 그대로 하면 더러우니까 다른 회사들처럼 중심 내용을 제목으로").
    message = 지금까지 사용자가 한 말(최근 3개, 첨부 이름 포함), reply = 마지막 답 앞부분.
    인사·잡담뿐이면 빈 글(페이지는 '새 대화'로 두고 다음 주고받기에서 다시). 질문이 모호하면('풀어봐') 답·첨부에서 주제를 찾는다."""
    message, reply = message.strip()[:1500], reply.strip()[:700]
    if not message:
        return ""
    if HANGUL.search(message):
        how = ("in Korean, as a short noun phrase of about 6-16 Korean characters, not a sentence. "
               "Examples: '내일 서울 강수 예보', '분수 지수 계산', '파이썬으로 1~100 합계 구하기', '이순신 장군 생애', '삼차함수 f(3) 구하기'")
    else:
        how = ("in the same language as the user's messages, at most 6 words. "
               "Examples: 'Sum of 1 to 100 in Python', 'Tomorrow's Rain Forecast in Seoul'")
    system = ("You name conversations for a chat history list. Read what the user said (and the start of the reply) and write a title "
              "that tells the main topic at a glance. If the user's words are vague (like 'solve it' or '풀어봐'), take the topic from the "
              "attachment names and the reply. Keep the key words that tell this chat apart (e.g. Python, Seoul, fractional exponents). "
              "Do not copy the message word for word. Write the title " + how + ". "
              "If the conversation is only greetings or small talk with no topic yet (e.g. hi, thanks, how are you), output exactly NONE. "
              "Output only the title on one line: no quotes, no period, no emoji, no markdown, no LaTeX, no 'Title:' prefix.")
    user = "USER SAID:\n" + message + ("\n\nREPLY (start):\n" + reply if reply else "")
    try:
        d = _call([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=40)
    except Busy:
        return ""
    t = (clean(d["choices"][0]["message"]["content"] or "").strip().split("\n") or [""])[0]
    t = re.sub(r"^(제목|title)\s*[:：]\s*", "", t, flags=re.I)
    t = re.sub(r"[*_`#$\\]", "", t).strip(" \"'“”‘’「」『』.。")
    if not t or re.fullmatch(r"none|없음", t, re.I):
        return ""
    return t[:40]


def _finish(msgs: list, d: dict, out: str) -> str:
    """답이 한도에 걸려 끊겼으면(finish_reason == "length") 끊긴 자리부터 이어 쓰게 해서 붙인다. 그래도 끊기면 끝에 표시."""
    fin = d["choices"][0].get("finish_reason") or ""
    n = 0
    while fin == "length" and out and n < CONTINUE_MAX:
        n += 1
        d = _call(msgs + [{"role": "assistant", "content": out}, {"role": "user", "content": CONTINUE_NOTE}])
        more = clean(d["choices"][0]["message"]["content"] or "")
        fin = d["choices"][0].get("finish_reason") or ""
        if not more:
            break
        out += more
    if n:
        log(f"이어 쓰기 {n}번 ({'그래도 끊김' if fin == 'length' else '끝까지'})")
    if fin == "length" and out:
        out += CUT_NOTE
    return out


# 설정 > 사용자 지침의 말투 (2026-11-10 사용자: "사용자 지침 탭 … 말투 선택 가능하게 하고 사용자 지침 넣는 기능도")
TONES = {
    "friendly": "말투 설정(앞의 말투 규칙보다 우선): 다정하고 편안하게. 한국어로 답할 땐 부드러운 해요체로, 공감 한마디를 곁들여도 좋아.",
    "casual": "말투 설정(앞의 말투 규칙보다 우선): 친한 친구처럼 반말로 편하게 말해(한국어일 때 '~야', '~해', '~지'). 다른 언어는 그 언어의 편한 말투로.",
    "formal": "말투 설정(앞의 말투 규칙보다 우선): 격식 있고 정중하게. 한국어로 답할 땐 '~습니다', '~입니다' 같은 합니다체로.",
    "brief": "말투 설정(앞의 말투 규칙보다 우선): 핵심만 아주 짧게. 인사·맞장구·덧붙임 없이 1~3문장으로(코드·풀이는 필요한 만큼).",
    "detailed": "말투 설정(앞의 말투 규칙보다 우선): 자세하고 친절하게. 배경·이유·예시까지 단계별로 충분히 설명해.",
}
MAX_INSTR = 1500


def reply(text: str, history: list, images=None, files=None, think: bool = False, memory=None, location=None,
          style: str = "", instructions: str = "", provider: str = "cf", older=None):
    """provider = "cf"(Gemma 4, Cloudflare) 또는 "groq"(gpt-oss-120b). older = 앞 대화 8개보다 오래된 사용자 말(줄임). 나머지는 _reply 설명 그대로."""
    tok = _provider.set("groq" if provider == "groq" and groq_ready() else "cf")
    try:
        return _reply(text, history, images, files, think, memory, location, style, instructions, older or [])
    finally:
        _provider.reset(tok)


def _reply(text: str, history: list, images=None, files=None, think: bool = False, memory=None, location=None,
           style: str = "", instructions: str = "", older=None):
    """(답, 덧붙일 정보 또는 None). 덧붙일 정보 = {"sources": [{title, url}], "searched": 검색어,
    "thought": {"seconds", "summary", "asked"}, "remember": [새로 기억할 한 줄]}.
    history는 app.py의 Turn 목록(role, content). images = data URL 목록, files = [{name, text}],
    think = 페이지의 '깊게 생각' 단추, memory = 이 브라우저에 저장된 사용자 기억(한 줄씩)."""
    if len(text) > MAX_TEXT_CHARS:  # 거의 없는 일: 책 한 권 넘는 글 → 앞·끝만 (가운데 생략)
        cut = len(text) - MAX_TEXT_CHARS
        text = text[:120000] + f"\n\n…(글이 너무 길어서 가운데 {cut:,}자는 빼고 보냄)…\n\n" + text[-80000:]
    asked, secs = think_request(text)
    think = bool(think or asked)
    if think and not asked and SMALL_TALK.match(text or ""):  # 단추만 켜져 있고 인사·맞장구면 생각 없이 바로
        think = False
    # 위치 공유 (2026-11-09): 페이지의 '위치' 단추를 켜면 대략 좌표(소수 둘째 자리 ≈ 1km)가 온다 → 동네 이름으로 바꿔 알려 준다(로그에는 안 남김)
    loc = None
    if location:
        try:
            la, lo = float(_field(location, "lat")), float(_field(location, "lon"))
            if -90 <= la <= 90 and -180 <= lo <= 180:
                loc = (round(la, 2), round(lo, 2), weather.reverse(la, lo))
        except (TypeError, ValueError):
            loc = None
    if loc:  # 위치를 알면 '지역 없는 날씨 질문 = 내 위치'로 (위치를 켰는데도 지역을 되묻던 일, 2026-11-09 PC 시험)
        wp = (f" (위치) 사용자가 위치 공유를 켰어: 지금 대략 {loc[2] or '알 수 없는 동네'} 근처에 있어(위도 {loc[0]}, 경도 {loc[1]}). "
              " 날씨·기온·비·눈·바람처럼 날씨를 물으면 웹 검색 말고 답 대신 한 줄로 <<weather: 지역 이름>> 만 써. "
              "지역을 말하지 않은 날씨 질문은 사용자가 있는 곳의 날씨를 묻는 것이니 반드시 <<weather: 내 위치>> 를 쓰고, 어느 지역인지 되묻지 마. "
              "앞 대화에서 다른 지역을 말했으면 그 지역을 써. 위치는 날씨·근처 장소 질문에만 쓰고, 묻지 않았는데 위치를 꺼내 말하지는 마.")
    else:
        wp = ""
    # 프롬프트 캐시(Groq·Cloudflare)는 앞부분이 똑같아야 맞는다 → 늘 같은 규칙(STATIC)을 앞에, 바뀌는 것(날짜·위치·기억·지침)은 뒤에
    system = (SYSTEM_PROMPT + (CALC_PROMPT if USE_CALC else "") + (SEARCH_PROMPT if USE_SEARCH else "") + WEATHER_PROMPT + REMEMBER_PROMPT
              + " ---- 이번 대화 정보: " + f"오늘은 한국 시각으로 {today_kst()}이야." + wp)
    if older:  # 앞 대화 8개보다 오래된 부분: 사용자가 했던 말을 짧게 (2026-11-11 '오래된 대화는 요약해서')
        system += " 이 대화의 앞부분에서 사용자가 했던 말(오래된 순, 줄임): " + " / ".join("«" + " ".join(str(o).split())[:160] + "»" for o in older[:12]) + ". 필요할 때 이어서 참고해."
    mem = [" ".join(str(m).split())[:150] for m in (memory or []) if str(m).strip()][:MAX_MEMORY]
    if mem:
        system += " 이 사용자에 대해 이 브라우저에 저장된 기억: " + " / ".join(mem) + ". 자연스럽게 활용하되 매번 꺼내 말하지는 마."
    tone = TONES.get((style or "").strip())
    if tone:
        system += " " + tone
    ins = " ".join(str(instructions or "").split())[:MAX_INSTR]
    if ins:  # 사용자가 설정에 직접 적은 지침 (안전·정직 규칙과 부딪히면 그 부분만 따르지 않음)
        system += (" 사용자가 설정의 '사용자 지침'에 직접 적어 둔 내용이야. 안전·정직 규칙과 부딪히지 않는 한 답할 때 꼭 따라: «" + ins + "»")
    groq = _provider.get() == "groq"
    if groq:
        system = system.replace("Gemma나 구글이 만든 모델이라고 소개하지 말고, 무슨 모델·기술로 움직이는지 직접 물을 때만 구글 Gemma 4를 바탕으로 한다고 말해.",
                                "다른 회사가 만든 모델이라고 소개하지 말고, 무슨 모델·기술로 움직이는지 직접 물을 때만 OpenAI의 공개 모델 gpt-oss-120b를 바탕으로 Groq에서 돌아간다고 말해.")
        # 말투는 강제하지 않는다 (2026-11-11 사용자: "해요체 절대 하지 마") — 설정의 말투를 고른 경우에만 그 말투
        system += " 수식은 $...$ 또는 $$...$$로 써."
    system_plain = system  # 생각하기가 실패해 생각 없이 다시 부를 때 쓰는 지시문 ('생각했다'는 말이 없음)
    # 빠른 생각 (2026-11-11 사용자: "생각만 1분은 좀 길어. 고작 26b인데") — Gemma의 생각 기능은 짧게 하라고 해도 한도(2,400~4,000토큰)를 다 써서
    # Cloudflare 속도(초당 26~46토큰)로 50~120초가 걸림 → '생각' 단추는 답 앞에 3~8줄 풀이를 먼저 쓰게 하고(보통 +3~8초), 그 풀이를 '생각 요약'으로 보여 준다.
    # "생각해 봐", "30초 동안 생각해" 처럼 직접 부탁하면 예전처럼 진짜 생각 기능.
    fast = think and not asked  # Groq도 빠른 생각: 호출 한 번·같은 설정이라 캐시가 맞아 무료 한도를 덜 씀 (2026-11-11 실서버: 진짜 생각+요약은 호출 2번·캐시 0)
    if fast:
        think = False
        system += (" 이번 답은 '생각' 단추를 켜고 만든 거야. 답을 쓰기 전에 맨 앞에 <<생각>> 한 줄을 쓰고, 그 아래에 이 질문을 어떻게 풀지 "
                   "3~8줄로 짧게 따져 봐(핵심 가정·계산·확인을 한 줄에 한 단계씩, 사용자가 쓴 언어로, 모두 합쳐 500자 안쪽). "
                   "그다음 <</생각>> 한 줄을 쓰고 이어서 사용자에게 보일 답을 써. 답에서는 생각 부분을 되풀이하지 말고 결론과 설명을 써. "
                   "도구(<<calc>>·<<search>>·<<weather>>)가 필요하면 <</생각>> 다음에 써. 몇 초 생각했는지 숫자는 지어내지 마.")
    if think:
        system += (" 이번 답은 실제로 '생각하기' 기능을 켜고 만들어" + (f"지고, 사용자에게는 실제로 {secs}초 넘게 지난 뒤에 보여져" if secs else "져") +
                   ". 답하거나 도구(<<calc>>·<<search>>)를 쓰기 전에 반드시 속으로 따져 보는 생각 과정을 먼저 거쳐. "
                   # 생각을 짧고 효율적으로 (2026-11-11: 어림 문제에서 생각 글 12,000자·94초)
                   "생각은 효율적으로 해: 꼭 필요한 단계만 차례로 따지고, 같은 계산·확인·다시 쓰기를 되풀이하지 마. "
                   "어림(페르미 추정) 문제는 핵심 가정 3~4개를 정해 곱하고 범위를 정한 뒤 바로 끝내. 쉬운 질문은 몇 줄만 생각해. "
                   "생각은 길어야 200단어 안팎으로 끝내고 곧바로 답을 써. "
                   "그러니 '생각해 봤어요'라고 말해도 사실이야. 답이 보일 때는 이미 생각이 끝난 뒤이니 '생각할게요'처럼 앞으로 하겠다고 말하지 마. "
                   "생각할 주제를 따로 주지 않았으면 지금까지의 대화나 네가 궁금한 것을 골라 생각하고, 무엇을 생각했는지 짧게 들려줘. 몇 초 생각했는지 숫자는 지어내지 마.")
    msgs = [{"role": "system", "content": system}]
    for h in history[-HISTORY_TURNS:]:
        msgs.append({"role": h.role, "content": h.content[:HISTORY_CHARS]})
    msgs.append({"role": "user", "content": _content(text, images, files)})
    t0 = time.time()
    # 생각하기는 시간을 정해도 토큰 한도를 줄이지 않는다 (예전엔 '10초'면 400토큰이라 생각+답이 잘림, 2026-11-09 사용자 신고).
    # 정한 시간보다 빨리 끝나면 아래에서 그 시간까지 기다리고, 늦게 끝나면 'n초 넘게' 생각한 것이라 사실과 어긋나지 않는다.
    try:
        d1 = _call(msgs, think=think)
    except Busy:
        # 그림 + 생각하기 요청에 Cloudflare가 내부 오류(500)를 내는 일이 있음(2026-11-09 실서버·PC) → 생각 없이라도 답한다(생각했다고는 말하지 않게)
        if not think:
            raise
        log("생각하기 요청 실패 → 생각 없이 다시")
        think = False
        msgs[0] = {"role": "system", "content": system_plain}
        d1 = _call(msgs)
    first = d1["choices"][0]["message"]
    raw = first.get("content") or ""
    reasoning = (first.get("reasoning_content") or "").strip() if think else ""
    fast_note = ""
    if fast:  # 빠른 생각: <<생각>> … <</생각>> 를 떼어 생각 요약으로
        m = re.match(r"\s*<<\s*생각\s*>>\s*(.*?)\s*<<\s*/\s*생각\s*>>\s*(.*)$", raw, re.S)
        if m:
            fast_note, raw = m.group(1).strip()[:1200], m.group(2)
        elif re.match(r"\s*<<\s*생각\s*>>", raw):  # 닫는 표시가 없으면 첫 빈 줄까지를 생각으로
            body = re.sub(r"^\s*<<\s*생각\s*>>\s*", "", raw)
            parts = body.split("\n\n", 1)
            fast_note, raw = parts[0].strip()[:1200], (parts[1] if len(parts) > 1 else parts[0])
    if think and not reasoning:  # 생각 글이 본문에 <think>…</think>로 섞여 오는 경우
        m = re.search(r"<think>(.*?)(?:</think>|$)", raw, re.S)
        reasoning = m.group(1).strip() if m else ""
    out = _finish(msgs, d1, clean(raw))
    summary = None
    if think and not out:  # 생각에 토큰을 다 써서 답이 비면: 생각 요약을 바탕으로 생각 없이 한 번 더
        summary = _think_summary(reasoning, text) if reasoning else ""
        msgs += [{"role": "assistant", "content": "(속으로 생각한 내용) " + (summary or "생각을 정리하는 중이었어.")},
                 {"role": "user", "content": "방금 생각한 것을 바탕으로 원래 말에 답해 줘. 생각 요약을 그대로 옮기지 말고, 원래 말에 대한 답을 해."}]
        d = _call(msgs)
        out = _finish(msgs, d, clean(d["choices"][0]["message"]["content"] or ""))
    extra, calcs, searched, weathered = {}, 0, False, False
    for _ in range(4):  # 도구는 한 질문에 날씨 1번 + 검색 1번 + 계산 2번까지
        mw = WEATHER_TAG.search(out) if not weathered else None
        ms = SEARCH_TAG.search(out) if USE_SEARCH and not searched else None
        mc = CALC_TAG.search(out) if USE_CALC and calcs < 2 else None
        if mw:
            weathered = True
            place = mw.group(1).strip()[:60]
            if MY_PLACE.match(place):
                note, src = weather.report(lat=loc[0], lon=loc[1]) if loc else (
                    "사용자의 위치를 몰라(위치 공유가 꺼져 있어). 어느 지역 날씨인지 물어보고, 입력칸의 '위치' 단추를 켜면 바로 알 수 있다고 알려 줘. 숫자는 지어내지 마.", None)
            else:
                note, src = weather.report(place)
            if src:
                extra.setdefault("sources", []).append(src)
            msgs += [{"role": "assistant", "content": mw.group(0)}, {"role": "user", "content": note}]
        elif ms:
            searched = True
            q = ms.group(1).strip()[:120]
            found = web_search.search(q)
            note = _search_note(q, found, len(extra.get("sources", [])))
            if found:
                extra.setdefault("sources", []).extend({"title": x["title"], "url": x["url"]} for x in found)
                extra["searched"] = q
            msgs += [{"role": "assistant", "content": ms.group(0)}, {"role": "user", "content": note}]
        elif mc:
            calcs += 1
            expr = mc.group(1).strip()
            try:
                note = f"계산 결과: {expr} = {calc_tool.calc(expr)}"
                log(f"계산기: {expr[:60]} 성공")
            except Exception as e:  # noqa: BLE001 — 계산기가 못 하는 식
                note = f"계산기가 이 식을 계산하지 못했어({e}). 계산 없이 아는 범위에서 답하고, 확실하지 않으면 정확히는 모른다고 해."
                log(f"계산기: {expr[:60]} 실패 {type(e).__name__}")
            msgs += [{"role": "assistant", "content": mc.group(0)},
                     {"role": "user", "content": note + CALC_ANSWER_NOTE}]
        else:
            break
        d = _call(msgs)
        out = _finish(msgs, d, clean(d["choices"][0]["message"]["content"] or ""))
    remember = [" ".join(m.split())[:150] for m in REMEMBER_TAG.findall(out)][:2]
    out = ANY_TAG.sub("", out).strip()  # 혹시 남은 도구 표시는 지움
    # 외국 글자가 섞이면 한 번 더(그래도 섞이면 그 글자만 지움) — 사용자가 그 글자(일본어·중국어·러시아어 등)로 쓴 경우는 그대로 둔다
    if STRAY.search(out) and not STRAY.search(text):
        out = STRAY.sub("", ANY_TAG.sub("", clean(_call(msgs)["choices"][0]["message"]["content"] or ""))).strip()
    if remember:
        extra["remember"] = remember
    if fast:  # 빠른 생각: 풀이를 그대로 요약으로 (요약을 만드는 호출 한 번을 아낌)
        extra["thought"] = {"seconds": max(1, round(time.time() - t0)), "summary": clean(fast_note)[:800] if fast_note else "", "asked": 0}
        log(f"빠른 생각: {round(time.time() - t0)}초 (풀이 {len(fast_note)}자)")
    if think:
        if summary is None:
            summary = _think_summary(reasoning, text) if reasoning else ""
        left = secs - (time.time() - t0)
        if left > 0:  # 사용자가 정한 시간까지 실제로 기다린다 ("10초 동안 생각해 봤어요"가 사실이 되게)
            time.sleep(left)
        # 실제로 생각 글이 있었거나 정한 시간만큼 기다렸을 때만 '생각함'으로 알린다 (모델이 생각을 건너뛰면 생각했다고 하지 않음)
        if reasoning or secs:
            extra["thought"] = {"seconds": round(time.time() - t0), "summary": summary, "asked": secs}
        log(f"생각하기: {round(time.time() - t0)}초 (정한 시간 {secs}초, 생각 글 {len(reasoning)}자)")
    return (out or "..."), (extra or None)


def reply_old_wiki(text: str, history: list):
    """예전 위키 참고 자료 방식 (USE_WIKI_CONTEXT=True일 때만 쓰던 것, 지금은 안 씀 — 기록용)."""
    ctx = wiki_context(text)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-HISTORY_TURNS:]:
        msgs.append({"role": h.role, "content": h.content[:HISTORY_CHARS]})
    if ctx:
        # 괄호 안(생몰년·높이 같은 숫자)을 지우면 답이 사라진다 → wiki_search._clean은 쓰지 않고 공백만 정리
        # (2026-11-08 사용자: "위키백과 돌려서 오히려 멍청하다") → 자료는 참고만, 관련 없으면 무시하고 아는 대로. 숫자·날짜만 조심.
        docs = "\n\n".join(f"[참고 자료: 위키백과 「{t}」]\n{re.sub(' +', ' ', e.replace(chr(160), ' '))[:800]}" for t, e, _ in ctx)
        msgs.append({"role": "user", "content": f"{docs}\n\n위 자료는 참고용이야. 질문과 관련 있으면 활용하고, 관련 없거나 답이 없으면 자료 이야기는 하지 말고 "
                                                f"네가 아는 대로 자연스럽게 답해. 다만 숫자·날짜는 자료에 있거나 확실히 알 때만 말하고, 확실하지 않으면 정확히는 모른다고 해.\n"
                                                f"질문: {text}"})
    else:
        msgs.append({"role": "user", "content": text})
    d = _call(msgs)
    out = clean(d["choices"][0]["message"]["content"] or "")
    for _ in range(2 if USE_CALC else 0):  # 계산기는 한 질문에 최대 2번
        m = CALC_TAG.search(out)
        if not m:
            break
        expr = m.group(1).strip()
        try:
            note = f"계산 결과: {expr} = {calc_tool.calc(expr)}"
            log(f"계산기: {expr[:60]} 성공")
        except Exception as e:  # noqa: BLE001 — 계산기가 못 하는 식
            note = f"계산기가 이 식을 계산하지 못했어({e}). 계산 없이 아는 범위에서 답하고, 확실하지 않으면 정확히는 모른다고 해."
            log(f"계산기: {expr[:60]} 실패 {type(e).__name__}")
        msgs += [{"role": "assistant", "content": m.group(0)},
                 {"role": "user", "content": note + "\n이 결과로 원래 질문에 자연스럽게 답해. 숫자는 계산 결과에 적힌 그대로 옮겨 적고 쉼표를 새로 넣지 마"
                                                    "(긴 수에 쉼표를 넣다가 끝자리가 망가진 일이 있음). <<calc>>는 다시 쓰지 마."}]
        out = clean(_call(msgs)["choices"][0]["message"]["content"] or "")
    out = CALC_TAG.sub("", out).strip()  # 혹시 남은 표시는 지움
    if STRAY.search(out):  # 외국 글자가 섞이면 한 번 더(그래도 섞이면 그 글자만 지움)
        out = STRAY.sub("", clean(_call(msgs)["choices"][0]["message"]["content"] or "")).strip()
    return (out or "..."), (ctx[0][2] if ctx else None)


if ready():
    threading.Thread(target=_refresh_usage_loop, daemon=True).start()
