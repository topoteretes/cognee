<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  Cognee - 정확하고 지속적인 AI 메모리

  <p align="center">
  <a href="https://www.youtube.com/watch?v=1bezuvLwJmw&t=2s">데모</a>
  .
  <a href="https://docs.cognee.ai/">문서</a>
  .
  <a href="https://cognee.ai">더 알아보기</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Discord 참여</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">r/AIMemory 참여</a>
  .
  <a href="https://github.com/topoteretes/cognee-community">커뮤니티 플러그인 & 애드온</a>
  </p>


  [![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)
  [![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/stargazers/)
  [![GitHub commits](https://badgen.net/github/commits/topoteretes/cognee)](https://GitHub.com/topoteretes/cognee/commit/)
  [![GitHub tag](https://badgen.net/github/tag/topoteretes/cognee)](https://github.com/topoteretes/cognee/tags/)
  [![Downloads](https://static.pepy.tech/badge/cognee)](https://pepy.tech/project/cognee)
  [![License](https://img.shields.io/github/license/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/blob/main/LICENSE)
  [![Contributors](https://img.shields.io/github/contributors/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/graphs/contributors)
  <a href="https://github.com/sponsors/topoteretes"><img src="https://img.shields.io/badge/Sponsor-❤️-ff69b4.svg" alt="Sponsor"></a>

<p>
  <a href="https://www.producthunt.com/posts/cognee?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-cognee" target="_blank" style="display:inline-block; margin-right:10px;">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=946346&theme=light&period=daily&t=1744472480704" alt="cognee - Memory&#0032;for&#0032;AI&#0032;Agents&#0032;&#0032;in&#0032;5&#0032;lines&#0032;of&#0032;code | Product Hunt" width="250" height="54" />
  </a>

  <a href="https://trendshift.io/repositories/13955" target="_blank" style="display:inline-block;">
    <img src="https://trendshift.io/api/badge/repositories/13955" alt="topoteretes%2Fcognee | Trendshift" width="250" height="55" />
  </a>
</p>

데이터를 사용하여 AI 에이전트를 위한 개인화되고 동적인 메모리를 구축하세요. Cognee를 사용하면 RAG를 확장 가능하고 모듈화된 ECL(추출[Extract], 인지화[Cognify], 로드[Load]) 파이프라인으로 대체할 수 있습니다.

  <p align="center">
  🌐 사용 가능한 언어
  :
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=fr">Français</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ja">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=zh">中文</a>
  </p>


<div style="text-align: center">
  <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/main/assets/cognee_benefits.png" alt="Why cognee?" width="50%" />
</div>
</div>

## Cognee 소개

Cognee는 원시 데이터를 에이전트를 위한 지속적이고 동적인 AI 메모리로 변환하는 오픈 소스 도구이자 플랫폼입니다. 벡터 검색과 그래프 데이터베이스를 결합하여 문서를 의미적으로 검색 가능하게 하고 관계별로 연결합니다.

Cognee는 두 가지 방식으로 사용할 수 있습니다.

1. [Cognee 오픈 소스 (셀프 호스팅)](https://docs.cognee.ai/getting-started/installation): 기본적으로 모든 데이터를 로컬에 저장합니다.
2. [Cognee Cloud (관리형)](https://platform.cognee.ai/): 관리형 인프라에서 동일한 OSS 스택을 사용하여 더 쉽게 개발하고 프로덕션화할 수 있습니다.

### Cognee 오픈 소스 (셀프 호스팅):

- 과거 대화, 파일, 이미지, 오디오 스크립트 등 모든 유형의 데이터를 상호 연결
- 기존 RAG 시스템을 그래프와 벡터 기반의 통합 메모리 계층으로 대체
- 품질과 정밀도를 향상시키면서 개발자 노력과 인프라 비용 절감
- 30개 이상의 데이터 소스에서 데이터를 수집할 수 있는 Pythonic 데이터 파이프라인 제공
- 사용자 정의 작업, 모듈식 파이프라인, 내장 검색 엔드포인트를 통한 높은 사용자 정의 가능성 제공

### Cognee Cloud (관리형):
- 호스팅된 웹 UI 대시보드
- 자동 버전 업데이트
- 리소스 사용량 분석
- GDPR 준수, 엔터프라이즈급 보안

## 기본 사용법 & 기능 가이드

자세한 내용은 [Colab 튜토리얼](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing)을 확인하세요.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing)

## 빠른 시작

단 몇 줄의 코드로 Cognee를 사용해 보세요. 자세한 설정 및 구성은 [Cognee 문서](https://docs.cognee.ai/getting-started/installation#environment-configuration)를 참조하세요.

### 필수 조건

- Python 3.10 ~ 3.13

### 1단계: Cognee 설치

**pip**, **poetry**, **uv** 또는 선호하는 Python 패키지 관리자를 사용하여 Cognee를 설치할 수 있습니다.

```bash
uv pip install cognee
```

### 2단계: LLM 구성
```python
import os
os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"
```
또는 [템플릿](https://github.com/topoteretes/cognee/blob/main/.env.template)을 사용하여 `.env` 파일을 생성하세요.

다른 LLM 공급자를 통합하려면 [LLM 공급자 문서](https://docs.cognee.ai/setup-configuration/llm-providers)를 참조하세요.

### 3단계: 파이프라인 실행

Cognee는 문서를 가져와 지식 그래프를 생성한 다음 결합된 관계를 기반으로 그래프를 쿼리합니다.

이제 최소한의 파이프라인을 실행해 보겠습니다.

```python
import cognee
import asyncio
from pprint import pprint


async def main():
    # Cognee에 텍스트 추가
    await cognee.add("Cognee turns documents into AI memory.")

    # 지식 그래프 생성
    await cognee.cognify()

    # 그래프에 메모리 알고리즘 추가
    await cognee.memify()

    # 지식 그래프 쿼리
    results = await cognee.search("What does Cognee do?")

    # 결과 표시
    for result in results:
        pprint(result)


if __name__ == '__main__':
    asyncio.run(main())

```

보시다시피, 출력은 이전에 Cognee에 저장한 문서에서 생성됩니다.

```bash
  Cognee turns documents into AI memory.
```

### Cognee CLI 사용

대안으로 다음 필수 명령으로 시작할 수 있습니다.

```bash
cognee-cli add "Cognee turns documents into AI memory."

cognee-cli cognify

cognee-cli search "What does Cognee do?"
cognee-cli delete --all

```

로컬 UI를 열려면 다음을 실행하세요.
```bash
cognee-cli -ui
```

## RTL and Mixed-Text Support

Cognee provides opt-in support for Right-to-Left (RTL) languages (Hebrew, Arabic) and mixed RTL/LTR text. It includes an automated normalization layer that detects visual-order text extraction (common in some PDF extractors) and converts it back to logical order for correct LLM processing.

### Setup

1. **Install with RTL dependencies**:
```bash
pip install "cognee[rtl]"
```

2. **Enable in Configuration**:
Set the environment variable or add it to your `.env` file:
```bash
ENABLE_RTL_SUPPORT=True
```

## 데모 및 예제

Cognee 작동 모습 확인:

### 지속적인 에이전트 메모리

[LangGraph 에이전트를 위한 Cognee 메모리](https://github.com/user-attachments/assets/e113b628-7212-4a2b-b288-0be39a93a1c3)

### 간단한 GraphRAG

[데모 보기](https://github.com/user-attachments/assets/f2186b2e-305a-42b0-9c2d-9f4473f15df8)

### Cognee와 Ollama

[데모 보기](https://github.com/user-attachments/assets/39672858-f774-4136-b957-1e2de67b8981)


## 커뮤니티 및 지원

### 기여하기
여러분들의 기여를 환영합니다! 여러분의 의견은 Cognee를 더 좋게 만드는 데 큰 도움이 됩니다. 시작하려면 [`CONTRIBUTING.md`](CONTRIBUTING.md)를 참조하세요.

### 규칙

우리는 포용적이고 존중하는 커뮤니티를 만들기 위해 노력하고 있습니다. 규칙은 [규칙 문서](https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md)를 확인해주세요.

## 연구 및 인용

최근 LLM 추론을 위한 지식 그래프 최적화에 관한 연구 논문을 발표했습니다.

```bibtex
@misc{markovic2025optimizinginterfaceknowledgegraphs,
      title={Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning},
      author={Vasilije Markovic and Lazar Obradovic and Laszlo Hajdu and Jovan Pavlovic},
      year={2025},
      eprint={2505.24478},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.24478},
}
```
