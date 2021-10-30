
# **6장 머신러닝 작업 흐름**



이전의 예에서는 시작할 레이블이 지정된 데이터 세트가 이미 있고 모델 교육을 즉시 시작할 수 있다고 가정했다. 현실 세계에서는 그렇지 않은 경우가 많다. 데이터셋에서 시작하는 것이 아니라 문제에서 시작하는 것입니다.

당신이 당신의 기계 학습 컨설팅 가게를 시작한다고 상상해 보세요. 회사를 설립하고, 멋진 웹사이트를 만들고, 네트워크에 알리죠. 프로젝트가 굴러 들어오기 시작합니다.
* "웨딩"에서 사진을 공유하는 소셜 네트워크 유형의 개인화된 사진 검색 엔진으로, 수동으로 태그할 필요 없이 결혼식 때 찍은 모든 사진을 검색합니다.
* 신생 채팅 앱의 게시물 중 스팸 및 불쾌한 텍스트 콘텐츠에 플래그를 표시합니다.
* 온라인 라디오 사용자를 위한 음악 추천 시스템 구축
* 전자상거래 웹 사이트에 대한 신용카드 사기 탐지.
* 주어진 시간에 주어진 사용자에게 어떤 광고를 제공할 것인지 결정하기 위해 광고 클릭률을 예측합니다.
* 쿠키 제조라인 컨베이어 벨트에 이상 쿠키 플래그 지정
* 위성 이미지를 사용하여 아직 알려지지 않은 고고학 유적지의 위치를 예측합니다.

**윤리 관련 참고 사항**

'얼굴 사진으로 누군가의 신뢰도를 평가하는 AI 구축'과 같이 윤리적으로 의심스러운 프로젝트를 제안받기도 한다. 무엇보다도, 이 프로젝트의 타당성은 의심스럽다: 왜 신뢰성이 누군가의 얼굴에 반영되는지는 명확하지 않다. 둘째, 그러한 일은 모든 종류의 윤리적인 문제에 대한 문을 열어줍니다. 이 작업에 대한 데이터 세트를 수집하는 것은 사진에 레이블을 붙이는 사람들의 편견과 편견을 기록하는 것과 같다. 이러한 데이터에 대해 훈련하는 모델은 이러한 동일한 편향을 단순히 블랙박스 알고리즘으로 인코딩하여 적법성을 얇게 보이게 하는 것입니다. 우리와 같이 대체로 기술 문맹인 사회에서, "AI 알고리즘은 이 사람을 신뢰할 수 없다고 말했다"는 이상하게도 "존 스미스는 이 사람을 신뢰할 수 없다고 말했다"보다 더 비중과 객관성을 가지고 있는 것으로 보인다. 당신의 모델은 실제 사람들의 삶에 부정적인 영향을 미치면서 인간의 판단의 가장 나쁜 면들을 세탁하고 작동시킬 것입니다.
기술은 결코 중립일 수 없다. 만약 여러분의 연구가 세상에 영향을 미친다면, 이 영향은 도덕적인 방향을 가집니다: 기술적인 선택 또한 윤리적 선택입니다. 당신의 작품이 뒷받침하기를 바라는 가치에 대해 항상 숙고하세요.


keras.datasets에서 올바른 데이터 세트를 가져와 일부 딥러닝 모델을 적합시킬 수 있다면 매우 편리할 것이다. 불행히도 현실 세계에서는 처음부터 다시 시작해야 할 것이다.


이 챕터에서는 위에 나열된 문제처럼 머신러닝에 접근하고 문제를 해결하는 데 사용할 수 있는 보편적인 단계별 청사진에 대해 알아보겠습니다. 이 템플릿은 4장과 5장에서 배운 모든 내용을 통합하고 통합하며, 다음 장에서 배울 내용을 파악할 수 있는 더 넓은 맥락을 제공합니다.


머신러닝의 보편적 워크플로는 크게 세 부분으로 구성된다.
* 과제 정의: 문제 영역과 고객이 질문한 내용을 뒷받침하는 비즈니스 논리를 이해합니다. 데이터 집합을 수집하고 데이터가 나타내는 바를 파악한 후 작업에서 성공을 측정하는 방법을 선택합니다.
* 모델 개발: 머신러닝 모델에서 데이터를 처리할 수 있도록 준비하고, 모델 평가 프로토콜과 간단한 기준선을 선택하여 과대적합이 가능한 일반화 능력을 갖춘 첫 번째 모델을 교육한 후, 가능한 한 최고의 일반화 성능을 달성할 때까지 모델을 정규화하고 조정합니다.
* 모델 배포: 이해 관계자에게 작업물을 전달하고, 웹 서버, 모바일 앱, 웹 페이지 또는 임베디드 장치로 모델을 전달하며, 실제 모델의 성능을 모니터링하고, 차세대 모델을 구축하는 데 필요한 데이터를 수집하기 시작합니다.

자, 이제 시작합시다.

## **6.1 문제 정의**

자신이 하고 있는 일의 맥락을 깊이 이해하지 않고는 좋은 일을 할 수 없습니다. 당신의 고객은 왜 이 문제를 해결하려고 합니까? 고객은 솔루션에서 어떤 가치를 창출할 수 있습니까? 모델이 어떻게 사용될 것이며 고객의 비즈니스 프로세스에 어떻게 부합될 것 같습니까? 어떤 종류의 데이터를 사용할 수 있거나 수집할 수 있습니까? 비즈니스 문제에 매핑할 수 있는 기계 학습 과제는 무엇입니까?

### **6.1.1 문제 구체화하기** 

기계 학습 문제를 구체화하려면 일반적으로 이해관계자들과의 많은 상세한 논의가 필요하다. 여기 여러분의 마음 위에 있어야 할 질문들이 있습니다. 
* 입력 데이터는 무엇입니까? 당신은 무엇을 예측하려고 하는 건가요? 예를 들어 영화 리뷰와 정서 주석이 모두 있는 경우에만 영화 리뷰의 감정을 분류하는 방법을 배울 수 있습니다. 이와 같이, 이 단계에서 데이터 가용성은 보통 제한 요소이다. 대부분의 경우 사용자가 직접 새 데이터셋을 수집하고 주석을 달아야 합니다(다음 섹션에서 설명). 
* 어떤 유형의 기계 학습 과제를 직면하고 있습니까? 이항 분류인가요? 다중 클래스 분류? 스칼라 회귀? 벡터 회귀? 멀티클래스, 멀티라벨 분류? 이미지 분할? 순위요? 클러스터링, 생성 또는 강화 학습과 같은 다른 것이 있습니까? 어떤 경우에는 머신러닝이 데이터를 이해하는 가장 좋은 방법이 아닐 수도 있고, 여러분은 평범한 구식 통계 분석과 같은 다른 것을 사용해야 합니다. 
 * 사진 검색 엔진 프로젝트는 멀티클래스, 멀티라벨 분류 작업입니다. 
 * 스팸 탐지 프로젝트는 이진 분류 작업입니다. "공격적인 내용"을 별도의 클래스로 설정하면 3원 분류 작업이 됩니다.
 * 음악 추천 엔진은 딥 러닝이 아닌 매트릭스 인수분해(협업 필터링)를 통해 더 잘 처리되는 것으로 나타났다. 
 * 카드사기탐지사업은이진분류과제입니다.
 * 클릭률 예측 프로젝트는 스칼라 회귀 작업입니다. 
 * 변칙 쿠키 탐지는 이진 분류 작업이지만, 원시 이미지에서 쿠키를 올바르게 잘라내기 위해서는 첫 번째 단계로 객체 탐지 모델이 필요합니다. "이상 검출"이라고 알려진 기계 학습 기법 세트는 이 설정에 적합하지 않습니다!
 * 위성 이미지에서 새로운 고고학 유적지를 찾는 프로젝트는 이미지 유사성 순위 매기기 작업입니다. 알려진 이미지와 가장 유사한 새로운 이미지를 검색해야 합니다. 
고고학 유적지
* 기존 솔루션은 어떤 모습입니까? 고객이 스팸 필터링 또는 신용 카드 사기 탐지를 처리하는 수작업 알고리즘을 이미 보유하고 있을 수 있습니다. 
if 진술. 아마도 한 사람이 현재 쿠키 공장에서 컨베이어 벨트를 모니터링하고 불량 쿠키를 수동으로 제거하거나 특정 아티스트를 좋아하는 사용자에게 보낼 노래 추천 재생 목록을 만드는 등 고려된 과정을 수작업으로 처리하고 있을 것이다. 어떤 시스템이 이미 구축되어 있는지, 어떻게 작동하는지 확실히 이해해야 합니다. 
* 처리해야 할 특별한 제약 조건이 있습니까? 예를 들어, 스팸 탐지 시스템을 구축하는 앱이 철저하게 종단 간 암호화되어 있으므로 스팸 탐지 모델은 최종 사용자의 전화기에서 작동해야 하며 외부 데이터 세트에 대해 교육을 받아야 합니다. 아마도 쿠키 필터링 모델은 원격 서버보다는 공장에서 임베디드 장치에서 실행되어야 하는 지연 시간 제약이 있을 것이다. 당신은 당신의 작품이 들어맞을 전체 맥락을 이해해야 합니다.


일단 조사를 마쳤으면, 여러분은 여러분의 입력이 무엇일지, 여러분의 목표가 무엇인지, 그리고 문제가 어떤 종류의 기계 학습 과제로 매핑되는지 알아야 합니다. 이 단계에서 여러분이 하는 가설에 유의하십시오. 
* 입력이 주어지면 목표를 예측할 수 있다는 가설을 세웁니다. 
* 사용 가능한(또는 곧 수집할) 데이터가 입력과 대상 간의 관계를 학습하는 데 충분한 정보를 제공한다는 가설을 세웁니다. 

작업 모형을 갖추기 전까지는 가설일 뿐이며 검증되거나 무효화되기를 기다리고 있습니다. 기계 학습으로 모든 문제를 해결할 수 있는 것은 아닙니다; 입력 X와 대상 Y의 예를 종합했다고 해서 X가 Y를 예측하기에 충분한 정보를 포함하고 있다는 것을 의미하지는 않습니다. 예를 들어, 최근 가격 이력을 감안할 때 주식 시장의 주식 움직임을 예측하려고 한다면, 가격 이력은 예측 정보를 많이 포함하고 있지 않기 때문에 성공할 가능성이 낮다.

### **6.1.2 데이터세트 모으기**

작업의 특성을 이해하고 입력과 대상이 무엇인지 알게 되면 대부분의 머신러닝 프로젝트에서 가장 힘들고 시간이 많이 걸리며 비용이 많이 드는 부분인 데이터 수집이 필요한 시점입니다. 
* 사진 검색 엔진 프로젝트에서는 먼저 분류하려는 레이블 세트를 선택해야 합니다. 10,000개의 공통 이미지 카테고리에 안착합니다. 그런 다음 이 세트의 레이블로 사용자가 업로드한 과거 이미지 수십만 개에 수동으로 태그를 지정해야 합니다.
* 채팅앱 스팸탐지 사업은 사용자대화가 종단간 암호화되어 있어 모델교육에 사용할 수 없습니다. 별도의 액세스 권한을 얻어야 합니다. 
수만 개의 공개 소셜 미디어 게시물에 대한 데이터 세트를 수동으로 태그하고 스팸, 불쾌감 또는 허용 가능한 태그를 지정합니다.
* 음악 추천 엔진은 사용자의 "좋아요"를 사용하면 됩니다. 새로운 데이터를 수집할 필요가 없습니다. 클릭률 예측 프로젝트도 마찬가지입니다. 과거 광고에서 클릭률에 대한 광범위한 기록이 있습니다.
* 쿠키 플래깅 모델의 경우, 수만 개의 이미지를 수집하기 위해 컨베이어 벨트 위에 카메라를 설치한 후 누군가가 수동으로 이 이미지에 레이블을 붙여야 합니다. 이 방법을 알고 있는 사람들은 현재 쿠키 공장에서 일하고 있습니다. 하지만 그다지 어려워 보이지는 않습니다. 여러분은 이 방법을 사용하도록 사람들을 훈련시킬 수 있어야 합니다.
* 위성사진 프로젝트에서는 고고학 팀이 기존 관심 장소의 데이터베이스를 수집해야 하며, 각 사이트에 대해 다른 기상 조건에서 찍은 기존 위성 사진을 찾아야 합니다. 좋은 모델을 얻으려면 수천 개의 다른 사이트가 필요할 것입니다.

5장에서 모델의 일반화 기능은 거의 전적으로 모델의 데이터 속성, 즉 보유한 데이터 포인트 수, 레이블의 안정성, 기능의 품질에 따라 학습된다는 것을 배웠습니다. 좋은 데이터 세트는 보살피고 투자할 가치가 있는 자산이다. 프로젝트에 50시간을 더 할애할 경우 증분 모델링 개선 사항을 검색하는 것보다 더 많은 데이터를 수집하는 것이 가장 효과적인 방법입니다. 

알고리즘보다 데이터가 더 중요하다는 지적은 구글 연구진이 2009년 발표한 '데이터의 불합리한 효과'(유진 위그너의 1960년 저서 '자연과학에서의 수학의 불합리한 효과'에 대한 리프)에서 가장 유명하다. 이는 딥러닝이 유행하기 전이지만, 놀랍게도 딥러닝의 부상은 데이터의 중요성만 더 크게 만들었다. 

지도 학습을 수행하는 경우 입력(예: 이미지)을 수집한 후에는 입력(예: 이미지 태그)에 대한 주석(예: 모델이 예측할 목표)이 필요합니다. 

때때로 음악 추천 작업이나 클릭률 예측 작업의 경우처럼 주석이 자동으로 검색될 수 있다. 하지만 종종 데이터에 주석을 직접 달아야 합니다. 이것은 노동력이 많이 드는 과정입니다.

**데이터 주석 인프라에 투자**

데이터 주석 공정에 따라 목표물의 품질이 결정되고, 이는 다시 모형의 품질을 결정합니다. 사용할 수 있는 옵션을 신중하게 고려하십시오. 
* 데이터에 주석을 직접 달아야 합니까?
* 라벨을 모으기 위해 Mechanical Turk와 같은 크라우드소싱 플랫폼을 사용해야 하는가?
* 데이터 레이블링 전문 회사의 서비스를 사용해야 합니까? 

아웃소싱은 잠재적으로 시간과 비용을 절약할 수 있지만 통제력을 빼앗습니다. Mechanical Turk와 같은 것을 사용하는 것은 비싸지 않고 잘 확장될 수 있지만, 여러분의 주석들은 꽤 시끄럽게 끝날지도 모릅니다. 

최상의 옵션을 선택하려면 작업 중인 제약 조건을 고려하십시오. 
* 데이터 라벨 작성자가 주제 전문가여야 합니까, 아니면 데이터에 주석을 달 수 있는 사람이 있습니까? 고양이 대 개 이미지 분류 문제의 라벨은 누구나 선택할 수 있지만 개 품종 분류 작업의 라벨은 전문 지식이 필요하다. 한편, 뼈 골절의 CT 스캔에 주석을 다는 것은 의학 학위를 필요로 합니다.
* 데이터에 주석을 달아야 하는 경우, 전문 지식을 교육할 수 있습니까? 그렇지 않다면, 관련 전문가와 어떻게 접촉할 수 있을까요?
* 전문가가 주석을 작성하는 방법을 알고 계십니까? 그렇지 않으면 데이터 세트를 블랙박스로 취급해야 하며 수동 기능 엔지니어링을 수행할 수 없습니다. 이는 중요하지는 않지만 제한적일 수 있습니다. 

데이터에 레이블을 붙이기로 결정한 경우 주석을 기록하는 데 사용할 소프트웨어를 스스로에게 물어보십시오. 당신이 직접 그 소프트웨어를 개발해야 할 수도 있습니다. 생산적인 데이터 주석 소프트웨어는 많은 시간을 절약할 수 있으므로 프로젝트 초기에 투자할 가치가 있습니다.

**비대표 데이터 주의**

머신러닝 모델은 이전에 본 것과 비슷한 입력만 이해할 수 있습니다. 따라서 교육에 사용되는 데이터는 생산 데이터의 "대표"여야 합니다. 이 문제는 모든 데이터 수집의 기반이 작동해야 합니다. 

사용자가 음식 이름을 알기 위해 사진을 찍을 수 있는 앱을 개발 중이라고 가정해 보자. 식도락가들에게 인기 있는 이미지 공유 소셜 네트워크의 사진을 사용하여 모델을 훈련시킵니다. 배포 시간이 다가오면 화난 사용자들의 피드백이 쏟아지기 시작합니다. 즉, 앱이 10점 만점에 8번 오답합니다. 무슨 일이야? 테스트 세트의 정확도는 90%를 훨씬 넘었습니다! 사용자가 업로드한 데이터를 빠르게 살펴보면 랜덤 스마트폰으로 찍은 랜덤 레스토랑의 모바일 사진 업로드가 전문가 품질의 밝은 식탐을 돋우는 사진과 전혀 다르다는 것을 알 수 있다. "당신의 교육 데이터는 생산 데이터를 대표하지 않았습니다." 기계학습 지옥에 온 걸 환영하는 최고의 죄악이지 

가능하면 모델이 사용될 환경에서 직접 데이터를 수집합니다. 영화 감상 분류 모델은 옐프 레스토랑 리뷰나 트위터 상태 업데이트가 아닌 새로운 IMDB 리뷰에 사용되어야 한다. 트윗의 감성을 평가하려면 프로덕션에서 예상하는 것과 유사한 사용자 집합에서 실제 트윗을 수집하고 주석을 다는 것부터 시작하십시오. 프로덕션 데이터에 대한 교육이 가능하지 않다면 교육 데이터와 프로덕션 데이터가 어떻게 다른지 완전히 이해하고 이러한 차이를 적극적으로 수정해야 합니다. 

당신이 알아야 할 관련 현상은 개념 드리프트입니다. 거의 모든 실제 문제, 특히 사용자 생성 데이터를 다루는 문제에서 개념 드리프트가 발생합니다. 개념 드리프트는 시간이 지남에 따라 생산 데이터의 속성이 변경되어 모델 정확도가 점차 저하될 때 발생합니다. 2013년에 훈련받은 음악 추천 엔진은 오늘날 그다지 효과적이지 않을 수 있다. 마찬가지로, 함께 작업한 IMDB 데이터 집합도 2011년에 수집되었으며, 이 데이터 집합으로 훈련된 모델은 시간이 지남에 따라 어휘, 표현 및 영화 장르가 발전함에 따라 2012년의 리뷰와 비교하여 2020년의 리뷰에서 제대로 수행되지 못할 가능성이 높다. 개념 표류는 신용 카드 사기 탐지와 같은 적대적 맥락에서 특히 극심하며, 사기 패턴은 실질적으로 매일 변한다. 빠른 개념 드리프트를 처리하려면 지속적인 데이터 수집, 주석 및 모델 재교육이 필요하다. 

기계 학습은 훈련 데이터에 존재하는 패턴을 암기하는 데만 사용될 수 있다는 것을 명심하십시오. 전에 본 것만 알아볼 수 있어요. 미래를 예측하기 위해 과거 데이터에 대해 훈련된 머신러닝을 사용하는 것은 미래가 과거와 같이 행동할 것이라는 가정을 하는 것이다. 그것은 종종 사실이 아니다.

**참고: 샘플링 편향의 문제**

비대표 데이터의 특히 교활하고 흔한 경우는 샘플링 편향이다. 표본 추출 치우침은 데이터 수집 공정이 예측하려는 것과 교호작용할 때 발생하며, 이로 인해 측정값의 편향이 발생합니다. 유명한 역사적 사례가 1948년 미국 대통령 선거에서 일어났다. 선거 당일 밤 시카고 트리뷴은 "듀이가 트루먼을 물리치다"라는 헤드라인을 실었다. 다음날 아침, 트루먼이 승자로 나타났다. Tribune지의 편집자는 전화 설문 조사 결과를 신뢰했지만 1948년의 전화 사용자들은 투표자 중 무작위로 추출한 대표적인 표본이 아니었다. 그들은 더 부유하고 보수적이며 공화당 후보인 듀이에게 투표할 가능성이 높았다. 요즘 모든 전화 조사는 샘플링 편향을 고려합니다. 그렇다고 해서 정치 여론조사에서 표본편향이 과거의 일이라는 것은 아니다. 그러나 1948년과는 달리, 여론 조사자들은 그것을 인식하고 그것을 바로잡기 위한 조치를 취한다.

### **6.1.3 데이터 이해**

데이터 세트를 블랙박스로 취급하는 것은 매우 나쁜 관행입니다. 모델을 교육하기 전에 데이터를 탐색하고 시각화하여 예측 가능한 요소에 대한 통찰력을 얻고(기능 엔지니어링에 정보를 제공) 잠재적인 문제를 선별해야 합니다. 
* 데이터에 이미지 또는 자연어 텍스트가 포함되어 있는 경우 샘플 몇 개(및 해당 레이블)를 직접 살펴보십시오.
* 데이터에 숫자 형상이 포함되어 있는 경우 형상 값의 히스토그램을 그래프로 표시하여 사용된 값의 범위와 다른 값의 빈도를 파악하는 것이 좋습니다.
* 데이터에 위치 정보가 포함되어 있으면 지도에 표시하십시오. 뚜렷한 패턴이 있나요?
* 일부 샘플에 일부 형상에 대한 결측값이 있습니까? 이 경우 데이터를 준비할 때 이 문제를 해결해야 합니다(다음 섹션에서 이 작업을 수행하는 방법에 대해 설명합니다.
* 분류 문제일 경우 데이터에 있는 각 클래스의 인스턴스 수를 인쇄합니다. 클래스가 대략적으로 동등하게 표현됩니까? 그렇지 않다면 이 불균형을 고려해야 합니다.
* "목표 누수"가 있는지 확인합니다. 데이터에 실운영에서 사용할 수 없는 대상에 대한 정보를 제공하는 기능이 있는지 확인합니다. 미래의 암 치료 여부를 예측하기 위해 의료기록에 대한 모델을 교육하고 있는데, 기록에 "이 사람이 암 진단을 받았다"는 특징이 포함되어 있다면, 여러분의 목표물이 인위적으로 여러분의 데이터에 유출되고 있는 것입니다. 데이터의 모든 기능이 운영 환경에서도 동일한 형태로 제공되는지 항상 자문해 보십시오.

### **6.1.4 성공의 척도를 선택한다.**

무언가를 통제하기 위해서는 그것을 관찰할 수 있어야 합니다. 프로젝트에서 성공을 거두려면 먼저 성공-정확성이 무엇인지 정의해야 합니다. 정확성과 기억력? 고객 유지율? 성공을 위한 여러분의 지표는 프로젝트 전반에 걸쳐 여러분이 하게 될 모든 기술적 선택을 안내할 것입니다. 고객의 비즈니스 성공과 같은 상위 수준의 목표와 직접 연계되어야 합니다. 

모든 클래스가 동일한 확률의 균형 분류 문제의 경우, "수신기 작동 특성 곡선(ROC AUC)" 아래의 정확도와 영역이 일반적인 지표이다. 클래스 불균형 문제, 순위 문제 또는 다중 레이블 분류의 경우 정밀도 및 호출뿐만 아니라 가중 형태의 정확도 또는 ROC AUC를 사용할 수 있습니다. 또한 성공을 측정하기 위해 자신만의 사용자 지정 메트릭을 정의해야 하는 경우도 흔합니다. 머신러닝 성공 메트릭의 다양성과 이러한 메트릭이 서로 다른 문제 영역과 어떻게 관련되는지 파악하려면 Kaggle(kaggle.com)에서 데이터 과학 대회를 검색하는 것이 도움이 됩니다. 다양한 문제와 평가 메트릭을 보여줍니다.

## **6.2 모델 개발** 
진행 상황을 어떻게 측정할 것인지 알고 나면 모델 개발을 시작할 수 있습니다. 대부분의 튜토리얼과 연구 프로젝트는 이것이 이미 수행된 것으로 가정되는 문제 정의 및 데이터 세트 수집을 건너뛰고 다른 사람이 처리하는 것으로 가정하는 모델 배치 및 유지 보수를 건너뛴다고 가정한다. 사실 모형 개발은 기계학습 워크플로우에서 한 단계일 뿐이고, 제가 보기에는 가장 어려운 것은 아닙니다. 머신러닝에서 가장 어려운 것은 틀에 박힌 문제와 데이터 수집, 주석 달기, 청소입니다. 그러니 힘내세요, 다음에 오는 것은 비교가 쉬울 거예요!

### **6.2.1 데이터 준비**
앞에서 배웠듯이 딥러닝 모델은 일반적으로 원시 데이터를 수집하지 않습니다. 데이터 전처리는 당면한 원시 데이터를 신경망에 더 잘 적응하도록 만드는 것을 목표로 한다. 여기에는 벡터화, 정규화 또는 결측값 처리가 포함됩니다. 많은 사전 처리 기술은 도메인마다 다릅니다(예: 텍스트 데이터 또는 이미지 데이터). 실제 예제에서 이러한 기술을 접할 때 다음 장에서 다루겠습니다. 지금은 모든 데이터 도메인에 공통적으로 적용되는 기본 사항에 대해 살펴보겠습니다.

**벡터화**

신경망의 모든 입력과 대상은 일반적으로 부동소수점 데이터의 텐서(또는 특정한 경우 정수나 문자열의 텐서)여야 한다. 소리, 이미지, 텍스트 등 필요한 데이터가 무엇이든 먼저 텐서로 전환해야 하며, 이 단계를 "데이터 벡터화"라고 합니다. 예를 들어, 4장의 이전 두 텍스트 분류 예에서는 정수 목록(단어 시퀀스를 나타냄)으로 표현된 텍스트에서 시작하여 원핫 인코딩을 사용하여 'float32' 데이터의 텐서로 변환했다. 숫자를 분류하고 집값을 예측한 예에서는 이미 데이터가 벡터화된 형태로 들어왔기 때문에 이 단계를 건너뛸 수 있었다.

**값 정규화**

2장의 MNIST 숫자 분류 예제에서는 0-255 범위의 정수로 인코딩된 영상 데이터에서 시작하여 그레이스케일 값을 인코딩했습니다. 이 데이터를 네트워크에 입력하기 전에 `float32`로 캐스팅하고 255로 나누어야 합니다. 
0-1 범위의 부동 소수점 값이 됩니다. 마찬가지로 집값을 예측할 때 부동소수점 값이 작고 정수 값이 상당히 큰 다양한 범위의 특성에서 시작했습니다. 이 데이터를 네트워크에 입력하기 전에 표준 편차가 1이고 평균이 0이 되도록 각 피쳐를 독립적으로 정규화해야 했습니다.
 
일반적으로 상대적으로 큰 값(예: 네트워크의 초기 가중치보다 훨씬 큰 여러 자리 정수)을 사용하는 신경망 데이터나 이기종 데이터(예: 한 특징이 0-1이고 다른 특징이 100-200인 데이터)에 입력하는 것은 안전하지 않다. 이렇게 하면 대규모 그라데이션 업데이트가 트리거되어 네트워크가 수렴되지 않을 수 있습니다. 네트워크에서 보다 쉽게 학습하려면 데이터에 다음과 같은 특성이 있어야 합니다. 

* 작은 값 사용 — 일반적으로 대부분의 값은 0-1 범위여야 합니다.
* 균일화 - 즉, 모든 형상이 거의 동일한 범위의 값을 가져야 합니다. 

또한 다음과 같은 엄격한 정규화 방법이 일반적이며 항상 필요한 것은 아니지만 도움이 될 수 있습니다(예: 숫자 분류 예제에서 이 작업을 수행하지는 않았습니다). 
* 평균이 0이 되도록 각 피쳐를 독립적으로 정규화합니다. 
* 표준 편차가 1이 되도록 각 형상을 독립적으로 정규화합니다.


```python
# NumPy array로 정규화 하기
x -= x.mean(axis=0)
x /= x.std(axis=0)
```

**결측값 처리**

때때로 데이터에 결측값이 있을 수 있습니다. 예를 들어, 집값의 예에서 첫 번째 특징(데이터의 지수 0 열)은 1인당 범죄율이었다. 만약 이 기능이 모든 샘플에서 사용할 수 없다면요? 그러면 교육 또는 검정 데이터에 결측값이 있게 됩니다. 

기능을 완전히 폐기할 수도 있지만 반드시 폐기할 필요는 없습니다. 
* 피쳐가 범주형인 경우 "값이 누락되었습니다"를 의미하는 새 범주를 만드는 것이 안전합니다. 모델은 대상에 대해 이것이 내포하는 의미를 자동으로 학습한다. 
* 형상이 숫자일 경우, "0"과 같은 임의의 값을 입력하지 마십시오. 형상에 의해 형성된 잠재 공간에 불연속성이 생성되어 해당 형상에 대해 훈련된 모델이 일반화하기가 더 어려울 수 있기 때문입니다. 대신 결측값을 데이터 집합의 기능에 대한 평균값 또는 중위값으로 바꾸는 것을 고려하십시오. 다른 형상의 값이 주어진 형상의 값을 예측하도록 모델을 교육할 수도 있습니다. 

테스트 데이터에 범주형 결측 기능이 있을 것으로 예상되지만 네트워크가 결측값 없이 데이터에 대해 학습된 경우, 네트워크는 결측값을 무시하는 방법을 배우지 않습니다. 이 경우 누락된 항목이 있는 교육 샘플을 인위적으로 생성해야 합니다. 일부 교육 샘플을 여러 번 복사하고 테스트 데이터에서 누락될 것으로 예상되는 범주형 기능 중 일부를 삭제해야 합니다.

### **6.2.2 평가규약 선택** 

이전 장에서 학습한 바와 같이 모델의 목적은 일반화를 달성하는 것이며, 모델 개발 프로세스 전반에 걸쳐 내릴 모든 모델링 결정은 일반화 성과를 측정하기 위한 검증 지표에 의해 안내됩니다. 검증 프로토콜의 목표는 실제 프로덕션 데이터에서 선택한 성공 지표(예: 정확도)를 정확하게 추정하는 것입니다. 그 과정의 신뢰성은 유용한 모델을 구축하는 데 매우 중요합니다. 

5장에서는 세 가지 일반적인 평가 프로토콜을 검토했습니다. 
* 홀드아웃 유효성 검사 세트 유지 - 데이터가 많을 때 사용하는 방법 
* K-폴드 교차 검증 수행 — 홀드아웃 유효성 검사가 신뢰할 수 없을 정도로 샘플 수가 적을 때 올바른 선택 
* 반복 K-폴드 검증 수행 - 데이터가 거의 없을 때 매우 정확한 모델 평가 수행 

이것들 중 하나만 골라. 대부분의 경우 첫 번째 방법은 충분히 효과가 있을 것입니다. 앞에서 배웠듯이, 항상 검증 세트의 대표성에 유의하고 교육 세트와 검증 세트 사이에 중복 샘플이 없도록 주의하십시오.

### **6.2.3 기준선 박치기** 
모형 자체에 대한 작업을 시작하면 5장에서 살펴본 것처럼 통계적 검정력을 달성하는 것이 초기 목표입니다. 즉, 간단한 기준선을 능가할 수 있는 작은 모형을 개발하는 것입니다. 

이 단계에서 가장 중요한 세 가지 사항은 다음과 같습니다. 
* 유용한 기능이 없는 기능을 걸러내고(기능 선택) 문제에 대한 지식을 활용하여 유용한 새로운 기능을 개발합니다. 
* 올바른 이전 아키텍처 선택: 어떤 유형의 모델 아키텍처를 사용할 예정입니까? 촘촘하게 연결된 네트워크, 컨브넷, 반복적인 신경 네트워크, 트랜스포머? 딥러닝은 과제를 위한 좋은 접근법인가요, 아니면 다른 것을 사용해야 하나요? 
* 충분한 교육 구성 선택—어떤 손실 기능을 사용해야 합니까? 배치 크기와 학습률은 어떻게 됩니까?

**참고: 올바른 손실 기능 선택**

문제에 대한 성공을 측정하는 메트릭에 대해 직접 최적화하지 못하는 경우가 많습니다. 메트릭을 손실 함수로 바꾸는 쉬운 방법이 없을 때도 있습니다. 결국 손실 함수는 데이터의 작은 배치(이상적으로 손실 함수는 단일 데이터 포인트만큼만 계산 가능해야 함)가 주어져야 하며, 역전파를 사용하여 네트워크를 훈련시킬 수 없어야 합니다. 예를 들어, 널리 사용되는 분류 지표 ROC AUC는 직접 최적화될 수 없습니다. 따라서 분류 작업에서 교차 엔트로피와 같은 ROC AUC의 프록시 메트릭에 최적화하는 것이 일반적이다. 일반적으로 교차 엔트로피가 낮을수록 ROC AUC가 더 높아지기를 바랄 수 있습니다. 

표 6.1은 몇 가지 일반적인 문제 유형에 대한 마지막 계층 활성화 및 손실 함수를 선택하는 데 도움이 될 수 있습니다. 

**표 6.1. 모형에 적합한 마지막 계층 활성화 및 손실 함수 선택**![캡처.PNG](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAe4AAAGyCAYAAAAiWyfaAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAIQaSURBVHhe7b2HmxbF26b9/S2SFHVBQEBglSRBeXkxgAmVpCgoSJIgSNIVE2sCF0RUogFzQkRBXlFERXRxFRDEBRERFNxFf4prfe9Z0/dY0/QzMz0zD/N0z3UeRx0zHZ4O1VV1Vd11V9X/54QQQgiRGSTcQgghRIaQcAshhBAZwgv3jBkzFBQUFBQUFEos3H777e7333/3gm1IuBUUFBQUFEo09O7d2/3v//2/vWAbXrj/1//6X+5//I//4XcIIYQQojT493//96qF+4wzzlBo4KFNmzZKCxkN+nYKCtkPLVu2dH/++afPyxJuhSrDf/kv/8WNHDlSaSGj4eabb9a3U1DIeLj++uvd//k//8fn5WoJ93/8x3+4f/u3f/P/i4bH//2//9f99//+3/3///N//k/3X//rf/X/i9Ln//2//+f+23/7b/5/MjqVMCFE9qAMlnCLaiPhzi4SbiHygYRbpELCnV0k3ELkAwm3SIWEO7tIuIXIBxJukYqGINz79u1zzz77rH/PY8eORXuzj4RbiHwg4RapqKlwf/zxx27dunXlYffu3dGR0oP0Pn36dHfKKae47777LtqbfSTcQuQDCbdIRU2Fe/Xq1X4Y2amnnuoWLlzovvjii+hIcdiwYYP77LPPoq30ULGQcAshShEJt0hFbUzlL774omvVqlW0VVzGjh3r3njjjWgrPRJuIUSpIuEWqTgZwv3333/7NLdy5Uq3bNky99577/l9Ib/88os/hhCRNnft2hUdce6rr77yM4Q98MADbtOmTT78+uuv0dHqUUi46fNeu3atv/eKFSvcjh07oiOu/F4E+93x48cr7A8XA6D74N5773Vz5szxx+wdf/75Z/fcc8/554effvrJLViwwL300kt+u6ZIuIXIBxJukYpiC/f27dvdhRde6C655BLfzzxp0iTXpEkTN2/evOgM5/bs2ePatm3rlixZ4jZv3uyPIbIIH4l58ODBrlGjRv46/E9I26eeJNw4rJ199tlu2LBhbvbs2W7AgAGuefPm7sCBA/74Pffc438zYsQIL8rwr3/9yz3yyCN+P6JJBQIBveOOO9xNN93k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)

대부분의 문제에서 시작할 수 있는 기존 템플릿이 있습니다. 스팸 탐지기, 음악 추천 엔진 또는 이미지 분류기를 만든 첫 번째 사람은 아닙니다. 선행 기술을 조사하여 작업에서 가장 잘 수행할 수 있는 기능 엔지니어링 기술과 모델 아키텍처를 식별하십시오. 

통계적 힘을 얻는 것이 항상 가능한 것은 아닙니다. 합리적인 아키텍처를 여러 번 시도해도 단순한 기준선을 넘을 수 없다면 입력 데이터에 질문에 대한 답이 없는 것일 수 있습니다. 두 가지 가설을 세우고 있다는 것을 기억하십시오. 
* 입력이 주어지면 출력을 예측할 수 있다는 가설을 세웁니다. 
* 사용 가능한 데이터가 입력과 출력 간의 관계를 학습하는 데 충분한 정보를 제공한다는 가설을 세웁니다. 
이 가설들이 거짓일 가능성이 높으며, 이 경우 당신은 처음부터 다시 시작해야 한다.

### **6.2.4 스케일업: 지나치게 적합한 모델 개발**

일단 통계적 힘을 가진 모델을 얻으면, 질문은 여러분의 모델이 충분히 강력하냐는 것입니다. 그것은 당면한 문제를 적절하게 모델링하기에 충분한 레이어와 파라미터를 가지고 있나요? 예를 들어, 로지스틱 회귀 분석 모형은 MNIST에 대한 통계적 검정력이 있지만 문제를 잘 해결하기에는 충분하지 않습니다. 머신러닝의 보편적인 장력은 최적화와 일반화 사이입니다. 이상적인 모델은 과소적합과 과적합, 과소용량과 과용량 사이의 경계에 서 있는 모델입니다. 이 국경이 어디에 있는지 알아내려면 먼저 국경을 넘어야 합니다.
 
얼마나 큰 모델이 필요한지 알아내려면, 당신은 지나치게 적합한 모델을 개발해야 합니다. 5장에서 학습한 바와 같이 이 작업은 매우 쉽습니다. 
1. 레이어를 추가합니다. 
1. 층을 크게 만드세요. 
1. 더 많은 시대를 위해 훈련하세요. 

교육 손실 및 검증 손실은 물론 관심 있는 메트릭에 대한 교육 및 검증 값도 항상 모니터링합니다. 검증 데이터에 대한 모형의 성능이 저하되기 시작하면 과적합이 이루어진 것입니다.

### **6.2.5 모델 정규화 및 조정**

일단 통계적 힘을 얻고 오버핏을 할 수 있게 되면, 여러분은 올바른 길을 가고 있다는 것을 알게 됩니다. 이때 일반화 성능을 최대화하는 것이 목표입니다. 

이 단계에서는 모델이 최대한 좋은 결과를 얻을 때까지 반복적으로 모델을 수정하고 교육하고 검증 데이터(현 시점에서 테스트 데이터가 아님)를 평가한 다음 다시 수정하고 반복합니다. 다음과 같은 몇 가지 방법을 시도해 보십시오. 
* 다양한 아키텍처를 시도하고 레이어를 추가 또는 제거합니다.
* 탈락 추가.
* 모형이 작은 경우 L1 또는 L2 정규화를 추가합니다.
* 최적의 구성을 찾기 위해 다양한 하이퍼 파라미터(예: 계층당 단위 수 또는 최적화 프로그램의 학습 속도)를 사용해 보십시오.
* 선택적으로 데이터 큐레이션 또는 기능 엔지니어링을 반복할 수 있습니다. 더 많은 데이터를 수집하고 주석을 달거나, 더 나은 기능을 개발하거나, 유용한 것으로 보이지 않는 기능을 제거합니다. 

Keras와 같은 "자동화된 하이퍼 파라미터 튜닝 소프트웨어"를 사용하여 이 작업의 상당 부분을 자동화할 수 있습니다.튜너. 13장에서 다루도록 하겠습니다. 

검증 프로세스의 피드백을 사용하여 모델을 조정할 때마다 검증 프로세스에 대한 정보가 모형에 유출됩니다. 몇 번만 반복해도 무해하지만, 여러 반복에 걸쳐 체계적으로 수행되면 검증 데이터에 대해 직접 교육을 받은 모델이 없음에도 불구하고 결국 모델이 검증 프로세스에 과도하게 적합하게 됩니다. 이것은 평가 과정의 신뢰성을 떨어뜨립니다. 

만족스러운 모델 구성을 개발했으면 사용 가능한 모든 데이터(교육 및 검증)에 대해 최종 생산 모델을 교육하고 테스트 세트에서 마지막으로 평가할 수 있습니다. 테스트 세트의 성능이 검증 데이터에서 측정된 성능보다 훨씬 더 나쁜 것으로 판명되면 이는 검증 절차를 신뢰할 수 없거나 모형의 매개 변수를 조정하는 동안 검증 데이터에 과적합하기 시작했음을 의미할 수 있습니다. 이 경우 K-폴드 반복 유효성 검사와 같은 보다 안정적인 평가 프로토콜로 전환할 수 있습니다.

## **6.3 모델 배포**

귀하의 모델은 테스트 세트에 대한 최종 평가를 성공적으로 마쳤습니다. 이제 배치하고 생산적인 수명을 시작할 준비가 되었습니다.

### **6.3.1 이해관계자에게 업무를 설명하고 기대치를 설정**

성공과 고객 신뢰는 지속적으로 고객의 기대에 부응하거나 그 이상을 달성하는 것입니다. 실제로 제공하는 시스템은 그 그림의 절반에 불과합니다. 나머지 절반은 출시 전 적절한 기대치를 설정하고 있다.

AI 시스템에 대한 비전문가들의 기대는 종종 비현실적이다. 예를 들어, 그들은 시스템이 과제를 "이해"하고 과제 맥락에서 인간과 같은 상식을 행사할 수 있다고 기대할 수 있다. 이 문제를 해결하려면 모형의 고장 모드의 몇 가지 예제를 보여 주는 것을 고려해야 합니다(예: 잘못 분류된 표본, 특히 잘못 분류된 표본이 놀라울 수 있는 표본 등).

또한 특히 이전에 사람이 처리했던 프로세스의 경우 인간 수준의 성능을 기대할 수 있습니다. 대부분의 머신러닝 모델은 인간이 만든 라벨에 근접하도록 (불완전하게) 훈련되었기 때문에 거의 도달하지 못한다. 모델 성능 기대치를 명확히 전달해야 합니다. "모델은 98%의 정확도를 가지고 있다"(대부분의 사람들이 정신적으로 최대 100%까지 반올림한다)와 같은 추상적인 문장을 사용하는 것을 피하고, 예를 들어, 잘못된 부정 비율과 잘못된 긍정 비율에 대해 이야기하는 것을 선호한다. "이러한 설정을 사용하면 부정 행위 탐지 모델은 5%의 거짓 음성 비율과 2.5%의 거짓 양성률을 갖습니다. 매일 평균 200건의 유효거래가 사기행위로 플래그가 지정돼 수작업 검토를 위해 발송되고, 평균 14건의 사기거래가 누락됐다. 평균 266건의 부정거래가 적발될 것이다." 모델의 성능 측정 기준을 비즈니스 목표와 명확하게 연관시킵니다.

또한 주요 시작 매개 변수(예: 트랜잭션에 플래그를 지정해야 하는 확률 임계값(임계값이 다르면 거짓 음수 및 거짓 긍정 비율이 다름)를 선택할 것인지 이해 관계자와 논의해야 합니다. 이러한 결정에는 비즈니스 맥락을 깊이 이해해야만 처리할 수 있는 트레이드오프가 포함됩니다.

### **6.3.2 추론 모델 발송**

머신러닝 프로젝트는 훈련된 모델을 저장할 수 있는 콜랩 노트북에 도착해도 끝나지 않습니다. 교육 중에 조작한 것과 동일한 파이썬 모델 객체를 프로덕션으로 넣는 경우는 거의 없습니다.

먼저 Python이 아닌 다른 것으로 모델을 내보내는 것이 좋습니다. 
* 운영 환경에서 Python을 전혀 지원하지 않을 수 있습니다(예를 들어, 모바일 앱이나 임베디드 시스템인 경우).
* 나머지 앱이 Python이 아닌 경우(JavaScript, C++ 등) 모델을 서비스하기 위해 Python을 사용하면 상당한 오버헤드가 발생할 수 있습니다. 

둘째, 생산 모델은 교육용이 아니라 예측(추론이라고 하는 단계) 출력에만 사용되므로 모델을 더 빠르게 만들고 메모리 공간을 줄일 수 있는 다양한 최적화를 수행할 수 있습니다. 

이제 사용 가능한 다양한 모델 구축 옵션을 간단히 살펴보겠습니다.

**REST API로 모델 배포**

이것은 아마도 모델을 제품으로 바꾸는 일반적인 방법일 것이다: 서버나 클라우드 인스턴스에 TensorFlow를 설치하고 REST API를 통해 모델의 예측을 쿼리한다. 플라스크(또는 다른 파이썬 웹)와 같은 것을 사용하여 자신만의 서빙 앱을 만들 수 있습니다. 
TensorFlow 서빙이라는 API로 모델을 제공하기 위해 TensorFlow의 자체 라이브러리를 사용합니다. TensorFlow 서빙(www.tensorflow.org/tfx/guide/serving)을 사용하여 Keras 모델을 몇 분 내에 배포할 수 있습니다. 

다음과 같은 경우 이 배포 설정을 사용해야 합니다. 
* 모델의 예측을 사용할 애플리케이션은 인터넷에 안정적으로 액세스할 수 있습니다(확실히). 예를 들어 응용 프로그램이 모바일 응용 프로그램인 경우 원격 API에서 예측을 제공한다는 것은 응용 프로그램을 비행기 모드나 저연결 환경에서 사용할 수 없음을 의미합니다.
* 응용 프로그램에는 엄격한 대기 시간 요구사항이 없습니다. 요청, 추론 및 응답 왕복에는 일반적으로 약 500ms가 소요됩니다. 
* 추론을 위해 전송된 입력 데이터는 그다지 민감하지 않습니다. 즉, 데이터는 모델에서 확인할 필요가 있으므로 해독된 형태로 서버에서 사용할 수 있어야 합니다(그러나 HTTP 요청 및 응답에는 SSL 암호화를 사용해야 함). 

예를 들어 이미지 검색 엔진 프로젝트, 음악 추천 시스템, 신용카드 사기 탐지 프로젝트, 위성 이미지 프로젝트는 모두 REST API를 통해 서비스하기에 적합합니다. 

REST API로 모델을 배포할 때 중요한 질문은 코드를 직접 호스팅할지 아니면 완전히 관리되는 타사 클라우드 서비스를 사용할지 여부입니다. 예를 들어 구글 제품인 클라우드 AI 플랫폼은 텐서플로우 모델을 구글 클라우드 스토리지(GCS)에 업로드하면 이를 쿼리할 수 있는 API 끝점을 제공한다. 일괄 처리 예측, 로드 밸런싱 및 확장과 같은 많은 실제적인 세부 사항을 처리합니다.

**장치에 모델 배포**

때로는 스마트폰, 로봇의 내장형 ARM CPU 또는 작은 장치의 마이크로컨트롤러 등 해당 애플리케이션을 실행하는 동일한 장치에 모델을 사용해야 할 수도 있습니다. 예를 들어, 여러분은 이미 카메라에서 직접 실행되는 작은 딥러닝 모델이었던, 여러분이 지목한 장면에서 사람과 얼굴을 자동으로 감지할 수 있는 카메라를 본 적이 있을 것입니다. 

다음과 같은 경우 이 설정을 사용해야 합니다. 
* 모델은 엄격한 지연 시간 제약이 있거나 연결성이 낮은 환경에서 실행해야 합니다. 몰입형 증강 현실 애플리케이션을 구축하는 경우 원격 서버를 쿼리하는 것은 실행 가능한 옵션이 아닙니다. 
* 모델은 대상 장치의 메모리 및 전력 제약 조건 하에서 실행될 수 있을 정도로 충분히 작게 만들 수 있습니다(TensorFlow Model Optimization Toolkit:(10)을 사용하여 이 문제를 해결할 수 있습니다).
* 가능한 한 높은 정확도를 얻는 것이 업무에 중요한 것은 아닙니다. 런타임 효율성과 정확성 사이에는 항상 균형이 있기 때문에 메모리 및 전력 제약으로 인해 대형 GPU에서 실행할 수 있는 최상의 모델보다 좋지 않은 모델을 제공해야 하는 경우가 많습니다. 
* 입력 데이터는 엄격하게 중요하므로 원격 서버에서 해독할 수 없습니다. 

예를 들어, 스팸 탐지 모델은 최종 사용자의 스마트폰에서 채팅 앱의 일부로 실행되어야 하는데, 이는 메시지가 종단 간 암호화되어 원격으로 호스팅된 모델에서 전혀 읽을 수 없기 때문이다. 마찬가지로 불량 쿠키 탐지 모델도 엄격한 대기 시간 제약이 있어 공장에서 실행해야 합니다. 다행히 이 경우 전력이나 공간 제약이 없어 GPU에서 실제로 모델을 실행할 수 있습니다. 

스마트폰이나 임베디드 장치에 Keras 모델을 배포하려면 TensorFlow Lite(www.tensorflow.org/lite)를 사용해야 합니다. ARM64 기반 컴퓨터, 라즈베리 파이 또는 특정 마이크로컨트롤러뿐만 아니라 Android 및 iOS 스마트폰에서 실행되는 효율적인 장치 딥러닝 추론을 위한 프레임워크입니다. Keras 모델을 TensorFlow Lite 형식으로 바로 전환할 수 있는 변환기가 포함되어 있습니다.

**브라우저에서 모델 배포**

딥 러닝은 브라우저 기반 또는 데스크톱 기반 자바스크립트 응용 프로그램에서 자주 사용됩니다. 응용 프로그램이 REST API를 통해 원격 모델을 쿼리하는 것이 보통 가능하지만 대신 사용자의 컴퓨터에서 직접 모델을 실행할 수 있는 주요 이점이 있을 수 있다. 

다음 경우에 이 설정을 사용합니다. 
* 컴퓨팅을 최종 사용자에게 오프로드하여 서버 비용을 크게 절감할 수 있습니다.
* 입력 데이터는 최종 사용자의 컴퓨터 또는 전화기에 남아 있어야 합니다. 예를 들어 스팸 탐지 프로젝트에서 채팅 앱의 웹 버전과 데스크톱 버전(JavaScript로 작성된 크로스 플랫폼 앱으로 구현됨)은 로컬에서 실행되는 모델을 사용해야 합니다.
* 애플리케이션에는 엄격한 지연 시간 제약이 있습니다. 최종 사용자의 노트북이나 스마트폰에서 실행되는 모델은 서버의 대형 GPU에서 실행되는 모델보다 속도가 느릴 수 있지만, 추가 100ms의 네트워크 왕복 시간이 없습니다.
* 모델이 다운로드되고 캐시된 후에도 연결 없이 계속 작동하려면 앱이 필요합니다. 

물론 모델이 사용자의 노트북이나 스마트폰의 CPU, GPU 또는 RAM을 독점하지 않을 정도로 작은 경우에만 이 옵션을 사용해야 합니다. 또한 전체 모델이 사용자의 장치에 다운로드되므로 모델에 대해 비밀로 유지할 필요가 없도록 해야 합니다. 훈련된 딥러닝 모델이 주어지면 일반적으로 훈련 데이터에 대한 일부 정보를 복구할 수 있다는 사실에 유의하십시오. 중요한 데이터에 대해 훈련된 모델은 공개하지 않는 것이 좋습니다. 

자바스크립트에서 모델을 배포하기 위해 텐서플로우 생태계는 TensorFlow.js(www.tensorflow.org/js),는 거의 모든 Keras API(원래 WebKeras라는 작업 이름으로 개발됨)뿐만 아니라 많은 하위 수준의 TensorFlow API를 구현한다. 저장된 Keras 모델을 TensorFlow.js로 쉽게 가져와 브라우저 기반 JavaScript 앱 또는 데스크톱 전자 앱의 일부로 쿼리할 수 있습니다.

**추론 모형 최적화**

사용 가능한 전력 및 메모리(스마트폰 및 임베디드 장치)에 엄격한 제약이 있는 환경이나 대기 시간이 짧은 애플리케이션에 배포할 때 추론을 위해 모델을 최적화하는 것이 특히 중요합니다. TensorFlow.js로 가져오거나 TensorFlow Lite로 내보내기 전에 항상 모델을 최적화해야 합니다. 

적용할 수 있는 두 가지 일반적인 최적화 기법이 있습니다. 
* 체중 가지치기: 체중 텐서의 모든 계수가 예측에 동일하게 기여하는 것은 아니다. 가장 중요한 항목만 유지하면 모델의 계층에서 매개변수 수를 크게 줄일 수 있습니다. 따라서 성능 메트릭에서 적은 비용으로 모델의 메모리 및 컴퓨팅 설치 공간을 줄일 수 있습니다. 적용할 가지치기 양을 조정하면 크기와 정확도 사이의 균형을 조정할 수 있습니다.
* 체중 정량화: 딥 러닝 모델은 단일 정밀 부동 소수점(float32 ) 가중치를 사용하여 훈련된다. 그러나 가중치를 8비트 부호 정수(int8)로 정량화하면 4배 작지만 원래 모델의 정확도에 가까운 추론 전용 모델을 얻을 수 있다. 

TensorFlow 생태계는 Keras API와 긴밀하게 통합된 가중치 정리 및 정량화 툴킷(www.tensorflow.org/model 최적화)을 포함한다.

### **6.3.3 야생에서 모델 모니터링**

추론 모델을 내보내고 이를 애플리케이션에 통합한 후 프로덕션 데이터에 대해 시운전을 수행했습니다. 모델이 예상대로 작동합니다. 유닛 테스트와 로깅 및 상태 모니터링 코드를 완벽하게 작성했습니다. 이제 큰 빨간색 버튼을 눌러 실운영에 투입할 차례입니다. 

심지어 이것이 끝이 아닙니다. 모델을 구축한 후에는 모델의 동작, 새 데이터에 대한 성능, 나머지 애플리케이션과의 상호 작용 및 궁극적으로 비즈니스 메트릭에 미치는 영향을 계속 모니터링해야 합니다. 
* 새로운 음악 추천 시스템을 구축한 후 온라인 라디오에 대한 사용자 참여가 증가했습니까, 아니면 감소했습니까? 새로운 클릭률 예측 모델로 전환한 후 평균 광고 클릭률이 증가했습니까? 랜덤 A/B 검정을 사용하여 모델 자체의 영향을 다른 변경으로부터 분리하는 것을 고려하십시오. 사례의 일부는 새 모형을 통과해야 하고 다른 제어 부분 집합은 이전 공정을 고수해야 합니다. 충분히 많은 사례가 처리되면, 두 사례의 결과 차이는 모델에 기인할 가능성이 높다.
* 가능하면 생산 데이터에 대한 모델의 예측에 대해 정기적인 수동 감사를 실시합니다. 일반적으로 데이터 주석과 동일한 인프라를 재사용할 수 있습니다. 생산 데이터의 일부를 수동으로 주석을 달도록 전송하고 모델의 예측값을 새 주석과 비교합니다. 예를 들어 이미지 검색 엔진과 잘못된 쿠키 플래그 지정 시스템에 대해 이 작업을 수행해야 합니다.
* 수동 감사가 불가능한 경우, 사용자 설문 조사와 같은 대안적인 평가 방법(예: 스팸 및 유해 콘텐츠 플래그 지정 시스템의 경우)을 고려하십시오.

### **6.3.4 모델 유지** 

마지막으로, 영원한 모델은 없습니다. 컨셉 드리프트에 대해 이미 배웠습니다. 시간이 지남에 따라 생산 데이터의 특성이 바뀌어 모델의 성능과 관련성이 점차 저하됩니다. 음악 추천 시스템의 수명이 몇 주 내로 계산됩니다. 신용카드 사기 탐지 시스템의 경우 며칠이 걸릴 수 있습니다. 이미지 검색 엔진을 위한 최고의 경우 2년입니다. 

모델이 출시되자마자 모델을 대체할 다음 세대를 교육할 준비를 해야 합니다. 이와 같이: 
* 생산 데이터의 변화를 주의합니다. 새로운 기능을 사용할 수 있습니까? 레이블 세트를 확장해야 합니까, 그렇지 않으면 편집해야 합니까? 
* 데이터를 계속 수집하고 주석을 달 수 있으며, 시간이 지남에 따라 주석 파이프라인을 계속 개선할 수 있습니다. 특히 현재 모형에 대해 분류하기 어려운 표본을 수집하는 데 특히 주의해야 합니다. 이러한 표본은 성능을 향상시키는 데 도움이 될 가능성이 높습니다. 

이것으로 머신러닝의 보편적인 워크플로를 마칩니다. 명심해야 할 사항이 많습니다. 전문가가 되기 위해서는 시간과 경험이 필요하지만 걱정하지 마세요, 여러분은 이미 몇 장 전보다 훨씬 더 현명해졌습니다. 이제 기계 학습 프로젝트에 수반되는 전체 스펙트럼이라는 큰 그림에 익숙해졌습니다. 이 책의 대부분은 모델 개발 부분에 초점을 맞추지만, 이제 전체 워크플로우의 일부분에 불과하다는 것을 알게 되었습니다. 항상 큰 그림을 명심하세요!

## **6.4 장 요약** 
* 새로운 머신러닝 프로젝트를 수행할 때 먼저 당면한 문제를 정의합니다. 
 * 해야 할 일의 광범위한 맥락을 이해합니다. 최종 목표는 무엇이며 제약은 무엇입니까?
 * 데이터 세트를 수집하고 주석을 달 수 있습니다. 데이터를 자세히 이해해야 합니다.
 * 문제에 대한 성공을 어떻게 측정할 것인지, 어떤 메트릭스에서 모니터링할 것인지 선택합니다. 
당신의 검증 데이터? 
* 문제를 이해하고 적절한 데이터 세트를 확보하면 모델을 개발합니다. 
 * 데이터를 준비합니다.
 * 평가 프로토콜을 선택하십시오. 홀드-아웃 검증? K-폴드 검증? 검증에 사용할 데이터 부분은 어느 정도입니까?
 * 통계적 능력을 달성하라: 단순한 기준선을 이긴다.
 * 스케일업: 오버핏 가능한 모델을 개발합니다.
 * 검증 데이터의 성능에 따라 모델을 정규화하고 하이퍼 파라미터를 조정합니다. 많은 머신러닝 연구는 이 단계에만 집중하는 경향이 있으며 큰 그림을 염두에 두고 있습니다.
* 모델이 준비되고 테스트 데이터에 대해 우수한 성능을 제공하면 이제 구현할 차례입니다. 
 * 먼저 이해 관계자들과 적절한 기대치를 설정했는지 확인합니다.
 * 추론을 위해 최종 모델을 최적화하고 웹 서버, 모바일, 브라우저, 임베디드 디바이스 등의 배포 환경에 모델을 전달합니다.
 * 모델의 생산 성능을 모니터링하고 데이터를 계속 수집하여 차세대 모델을 개발할 수 있습니다.
