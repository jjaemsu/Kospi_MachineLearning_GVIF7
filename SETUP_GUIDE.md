# 팀 프로젝트 필수 세팅 및 협업 가이드 (Team Setup Guide)

이 문서는 프로젝트에 처음 참여하는 팀원들을 위한 **초보자 맞춤형 가이드**입니다.
가상환경 세팅부터 깃허브 사용법까지 아래 순서대로 천천히 따라해 주세요!

---

## 1. 파이썬 가상환경(Virtual Environment) 세팅 🐍

팀원들 간에 파이썬 버전이나 설치된 패키지(예: pandas, xgboost 등) 버전이 다르면 코드가 실행되지 않는 문제가 발생합니다. 이를 방지하기 위해 **가상환경**을 만들어 우리 프로젝트만의 독립된 파이썬 공간을 사용합니다.

### Step 1. 가상환경 만들기
터미널(Terminal)을 열고 프로젝트 폴더 경로에서 아래 명령어를 입력하세요.
```bash
python -m venv venv
```
*(명령어를 실행하면 `venv`라는 이름의 폴더가 생성됩니다.)*

### Step 2. 가상환경 켜기 (활성화)
운영체제에 맞게 아래 명령어를 입력하여 가상환경을 켭니다.
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```
*(성공하면 터미널 입력창 왼쪽에 `(venv)`라는 글자가 나타납니다.)*

### Step 3. 필수 패키지 설치하기
가상환경이 켜진 상태에서, 프로젝트에 필요한 라이브러리들을 한 번에 설치합니다.
```bash
pip install -r requirements.txt
```

---

## 2. 초보자를 위한 Git & GitHub 사용법 🐙

깃(Git)은 코드를 저장하고 공유하는 '클라우드 장바구니'와 같습니다. 안전하게 코드를 작업하고 팀원들과 공유하는 방법을 알아봅시다.

### Step 1. 내 작업 공간(브랜치) 만들기
메인 원본 코드(`main`)를 직접 수정하지 않고, 나만의 작업 공간(브랜치)을 만들어서 안전하게 작업합니다.
```bash
# 내 기능 이름으로 브랜치 만들고 이동하기
git checkout -b feat/내이름-또는-기능이름
# 예시: git checkout -b feat/data-preprocessing
```

### Step 2. 작업 후 내 컴퓨터 장바구니에 담기 (Add & Commit)
코드를 수정했다면, 변경된 파일을 장바구니에 담고(`add`) 이름표를 붙여서 포장(`commit`)합니다.
```bash
# 변경된 모든 파일 장바구니에 담기
git add .

# 어떤 작업을 했는지 이름표 붙여서 포장하기
git commit -m "작업 내용 짧게 요약 (예: 데이터 전처리 코드 추가)"
```

### Step 3. 깃허브 서버로 전송하기 (Push)
포장한 코드를 깃허브 사이트로 보냅니다. (최초 1회만 아래 명령어를 사용합니다.)
```bash
git push -u origin feat/내이름-또는-기능이름
```

### Step 4. 메인 원본에 내 코드 합치기 요청 (Pull Request)
1. 깃허브 사이트(https://github.com/jjaemsu/Kospi_MachineLearning_GVIF7)에 접속합니다.
2. `Compare & pull request`라는 초록색 버튼을 클릭합니다.
3. 어떤 작업을 했는지 설명을 적고 `Create pull request`를 누르면, 팀원들이 확인 후 원본에 코드를 합쳐줍니다(Merge).

### Step 5. 팀원들의 최신 코드 받아오기 (Pull)
작업을 시작하기 전이나 다른 사람이 코드를 합쳤을 때, 내 컴퓨터의 코드를 최신 상태로 업데이트해야 합니다.
```bash
git checkout main
git pull origin main
```

---

## 3. 공통 AI 규칙 연동 방법 🤖

우리 팀은 **"사전 코드 작성 금지"**와 **"데이터 누수(Data Leakage) 방지"**를 위해 공통 AI 규칙(`AI_CONVENTIONS.md`)을 만들었습니다. 본인이 사용하는 AI 어시스턴트에 맞게 **최초 1회만** 아래 설정을 진행해 주세요.

- **Cursor 사용자:** 프로젝트 폴더 최상단에 `.cursorrules` 파일을 만들고 아래 내용을 복사해 넣습니다.
- **Cline 사용자:** 프로젝트 폴더 최상단에 `.clinerules` 파일을 만들고 아래 내용을 복사해 넣습니다.

**[복사할 내용]**
> "이 프로젝트에서 질문에 답하거나 코드를 제안할 때는, 반드시 프로젝트 루트 경로에 있는 `AI_CONVENTIONS.md` 파일을 먼저 읽고 그 안의 규칙을 엄격하게 준수해."