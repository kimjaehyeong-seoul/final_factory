"""
Google Sheets 유틸리티 모듈 - konlpy 기반 + 데이터 캐싱 시스템
"""

import streamlit as st
import pandas as pd
import gspread
import json
import os
import re
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

# ──────────── konlpy Okt 로드 ────────────
from konlpy.tag import Okt
okt = Okt()

# ================================
# 캐싱 시스템 설정
# ================================

# 캐시 디렉토리 생성
SHEETS_CACHE_DIR = Path("cache/sheets_data")
SHEETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SHEETS_CACHE_FILE = SHEETS_CACHE_DIR / "google_sheets_cache.pkl"
SHEETS_METADATA_FILE = SHEETS_CACHE_DIR / "sheets_metadata.json"

# ================================
# 설정
# ================================

# 전역 설정
try:
    SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
    if "SERVICE_ACCOUNT_INFO" in st.secrets:
        SERVICE_ACCOUNT_INFO = dict(st.secrets["SERVICE_ACCOUNT_INFO"])
    else:
        SERVICE_ACCOUNT_INFO = None
    SERVICE_ACCOUNT_FILE = None
except:
    SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1bCDIOk_QEf5Q56r0xmSmgTlGRvQB9Px3G-8HrUu0EiM")
    SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "diesel-practice-290305-76df92cfd2bd.json")
    SERVICE_ACCOUNT_INFO = None

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# ================================
# 서비스 계정 처리
# ================================

_service_account_cache = None

def get_service_account_info():
    """서비스 계정 정보 가져오기"""
    global _service_account_cache
    
    if _service_account_cache is not None:
        return _service_account_cache
    
    # 1순위: Streamlit secrets
    if SERVICE_ACCOUNT_INFO:
        _service_account_cache = SERVICE_ACCOUNT_INFO
        return SERVICE_ACCOUNT_INFO
    
    # 2순위: 로컬 파일
    possible_files = [
        "diesel-practice-290305-76df92cfd2bd.json",
        "diesel-practice-290305-ac532875b6e6.json",
        "service_account.json"
    ]
    
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    service_account_info = json.load(f)
                _service_account_cache = service_account_info
                print(f"✅ {filename}에서 서비스 계정 정보 로드 성공")
                return service_account_info
            except Exception as e:
                print(f"❌ {filename} 로드 실패: {e}")
    
    print("❌ 모든 서비스 계정 파일 로드 실패")
    return None

def get_google_sheets_client():
    """구글 시트 클라이언트 생성"""
    try:
        service_account_info = get_service_account_info()
        if service_account_info is None:
            return None
        
        gc = gspread.service_account_from_dict(service_account_info)
        return gc
    except Exception as e:
        print(f"구글 시트 클라이언트 생성 실패: {e}")
        return None

def get_connection_status():
    """연결 상태 확인"""
    try:
        gc = get_google_sheets_client()
        if gc is None:
            return {'connected': False, 'message': '서비스 계정 로드 실패'}
        
        sheet = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sheet.get_worksheet(0)
        
        # 연결 테스트
        records = worksheet.get_all_records()
        return {'connected': True, 'message': '정상 연결'}
    except Exception as e:
        return {'connected': False, 'message': str(e)}

# ================================
# 🚀 개선된 캐싱 시스템
# ================================

def get_sheets_last_modified():
    """구글 시트의 마지막 수정 시간 확인"""
    try:
        gc = get_google_sheets_client()
        if gc is None:
            return None
        
        sheet = gc.open_by_key(SPREADSHEET_ID)
        # 구글 시트 API로 마지막 수정 시간 확인
        # 여기서는 간단히 현재 시간을 사용 (실제 구현에서는 Drive API 사용)
        return datetime.now().timestamp()
    except:
        return None

def save_sheets_cache(df, metadata):
    """구글 시트 데이터를 캐시에 저장"""
    try:
        # 데이터 캐시 저장
        with open(SHEETS_CACHE_FILE, 'wb') as f:
            pickle.dump(df, f)
        
        # 메타데이터 저장
        with open(SHEETS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 구글 시트 데이터 캐시 저장 완료: {len(df)}행")
        return True
    except Exception as e:
        print(f"❌ 캐시 저장 실패: {e}")
        return False

def load_sheets_cache():
    """캐시에서 구글 시트 데이터 로드"""
    try:
        # 캐시 파일 존재 확인
        if not SHEETS_CACHE_FILE.exists() or not SHEETS_METADATA_FILE.exists():
            return None, None
        
        # 메타데이터 로드
        with open(SHEETS_METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 데이터 로드
        with open(SHEETS_CACHE_FILE, 'rb') as f:
            df = pickle.load(f)
        
        print(f"✅ 캐시에서 데이터 로드 성공: {len(df)}행")
        return df, metadata
    except Exception as e:
        print(f"❌ 캐시 로드 실패: {e}")
        return None, None

def is_cache_valid(metadata, max_age_hours=1):
    """캐시 유효성 검사"""
    if not metadata:
        return False
    
    # 시간 기반 캐시 유효성
    cache_time = metadata.get('timestamp', 0)
    current_time = time.time()
    
    if current_time - cache_time > max_age_hours * 3600:
        print(f"⏰ 캐시가 {max_age_hours}시간을 초과하여 만료됨")
        return False
    
    return True

def load_data_from_google_sheets():
    """구글 시트에서 직접 데이터 로드 (캐시 없이)"""
    try:
        gc = get_google_sheets_client()
        if gc is None:
            return pd.DataFrame()
        
        sheet = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sheet.get_worksheet(0)
        
        data = worksheet.get_all_records()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        print(f"✅ 구글 시트에서 직접 로드 성공: {len(df)}행, {len(df.columns)}열")
        return df
    except Exception as e:
        print(f"❌ 구글 시트 로드 실패: {e}")
        return pd.DataFrame()

def load_data_from_google_sheets_cached(force_refresh=False, max_age_hours=1):
    """
    캐싱된 구글 시트 데이터 로드 (증분 업데이트 지원)
    
    Args:
        force_refresh: 강제 새로고침 여부
        max_age_hours: 캐시 유효 시간 (시간)
    
    Returns:
        DataFrame: 구글 시트 데이터
    """
    try:
        # 강제 새로고침이 아닌 경우 캐시 확인
        if not force_refresh:
            cached_df, cached_metadata = load_sheets_cache()
            if cached_df is not None and is_cache_valid(cached_metadata, max_age_hours):
                print("✅ 유효한 캐시 데이터 사용")
                return cached_df
        
        # 캐시가 없거나 만료된 경우 새로 로드
        print("🔄 구글 시트에서 새 데이터 로드 중...")
        new_df = load_data_from_google_sheets()
        
        if new_df.empty:
            # 새 데이터 로드 실패시 캐시 데이터라도 반환
            cached_df, _ = load_sheets_cache()
            if cached_df is not None:
                st.warning("⚠️ 새 데이터 로드 실패, 캐시 데이터 사용")
                return cached_df
            return pd.DataFrame()
        
        # 새 데이터를 캐시에 저장
        metadata = {
            'timestamp': time.time(),
            'row_count': len(new_df),
            'column_count': len(new_df.columns),
            'last_updated': datetime.now().isoformat()
        }
        
        save_sheets_cache(new_df, metadata)
        return new_df
        
    except Exception as e:
        print(f"❌ 캐싱된 데이터 로드 실패: {e}")
        # 오류 발생시 캐시 데이터라도 반환
        cached_df, _ = load_sheets_cache()
        if cached_df is not None:
            st.error(f"데이터 로드 오류 발생, 캐시 데이터 사용: {e}")
            return cached_df
        return pd.DataFrame()

def test_connection():
    """연결 테스트"""
    status = get_connection_status()
    return status['connected']

# ================================
# 가중치 계산 함수들 (편차 축소 & 다양성 확보)
# ================================

def get_major_media_list():
    """중요 매체 목록 정의"""
    return {
        '조선일보', '동아일보', '중앙일보', '한국경제', '매일경제', '서울경제',
        'KBS', 'MBC', 'SBS', 'YTN', 'JTBC',
        '한국경제신문', '파이낸셜뉴스', '이데일리', '뉴시스', '연합뉴스'
    }

def calculate_media_weight(media_name):
    """매체별 가중치 계산 (편차 축소 버전)"""
    if pd.isna(media_name):
        return 0.05
    
    media_name = str(media_name).strip()
    major_media = get_major_media_list()
    
    # 주요 일간지
    if media_name in ['조선일보', '동아일보', '중앙일보', '한국경제', '매일경제']:
        return 0.12
    
    # 지상파/종편
    elif media_name in ['KBS', 'MBC', 'SBS', 'YTN', 'JTBC']:
        return 0.10
    
    # 기타 주요 매체
    elif media_name in major_media:
        return 0.08
    
    # 일반 매체
    else:
        return 0.05

def calculate_time_weight(date_str):
    """시간 가중치 계산 (최신일수록 높은 가중치)"""
    try:
        if pd.isna(date_str):
            return 0.5
        
        date_str = str(date_str).strip()
        
        # 현재 날짜 정보
        now = datetime.now()
        current_month = now.month
        current_day = now.day
        
        # 월일 패턴 추출
        month_day_pattern = r'(\d{1,2})월(\d{1,2})일'
        match = re.search(month_day_pattern, date_str)
        
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            
            # 월 차이 계산
            month_diff = abs(current_month - month)
            if month_diff > 6:  # 연도 차이 고려
                month_diff = 12 - month_diff
            
            # 가중치 계산 (최신일수록 높음)
            if month_diff == 0:
                # 같은 달이면 일 차이 고려
                day_diff = abs(current_day - day)
                if day_diff <= 3:
                    return 1.0  # 최근 3일
                elif day_diff <= 7:
                    return 0.9  # 최근 1주
                elif day_diff <= 14:
                    return 0.8  # 최근 2주
                else:
                    return 0.7  # 같은 달
            elif month_diff == 1:
                return 0.6  # 지난 달
            elif month_diff == 2:
                return 0.5  # 2달 전
            else:
                return 0.4  # 그 이전
        
        return 0.5  # 기본값
    except:
        return 0.5

def calculate_article_weight(row):
    """기사 가중치 계산 (편차 축소 버전)"""
    try:
        weight = 0.0
        
        # 1. 구분(일반/단독) 가중치
        distinction = row.get('구분(일반/단독)', '')
        if pd.notna(distinction):
            distinction = str(distinction).strip()
            if distinction == '단독':
                weight += 0.15
            else:
                weight += 0.08
        else:
            weight += 0.08
        
        # 2. 매체 가중치
        media = row.get('매체', '')
        if pd.notna(media):
            weight += calculate_media_weight(media)
        else:
            weight += 0.05
        
        # 3. 지면 가중치
        page = row.get('지면', '')
        if pd.notna(page):
            page = str(page).strip()
            if page == '온라인' or page == '':
                weight += 0.03
            else:
                weight += 0.08
            
            # 추가 세부 위치 가중치 (편차 축소)
            page_lower = page.lower()
            extra_weight = 0.0
            
            if 'TOP' in page.upper():
                extra_weight += 0.05
            elif '1단' in page:
                extra_weight += 0.03
            
            # 면수 분석
            if '면' in page_lower:
                page_part = page_lower.split('면')[0]
                # 숫자 추출
                numbers = re.findall(r'\d+', page_part)
                if numbers:
                    page_num = int(numbers[0])
                    # 앞면일수록 높은 가중치 (최대 0.12)
                    extra_weight += max(0.0, 0.12 - (page_num - 1) * 0.015)
            
            # 최대 추가 가중치 제한
            weight += min(extra_weight, 0.2)
        else:
            weight += 0.03
        
        # 4. 시간 가중치 추가
        date_str = row.get('보도날짜', '')
        time_weight = calculate_time_weight(date_str)
        weight += time_weight * 0.2  # 0.2배 반영
        
        # 최종 가중치 범위 제한 (0.2 ~ 0.8)
        normalized_weight = min(max(weight, 0.2), 0.8)
        return round(normalized_weight, 3)
    
    except Exception as e:
        print(f"가중치 계산 오류: {e}")
        return 0.5

# ================================
# konlpy 기반 업계 매핑 함수들 (확장)
# ================================

def load_industry_mapping_advanced():
    """확장된 업계 매핑 로드"""
    company_to_industry = {
        # 자동차
        '현대차': '자동차', '현대자동차': '자동차', '기아': '자동차', '기아차': '자동차',
        '삼성SDI': '자동차', '한온시스템': '자동차', '현대모비스': '자동차',
        '테슬라': '자동차', 'BMW': '자동차', '벤츠': '자동차', '아우디': '자동차',
        '포드': '자동차', '도요타': '자동차', '혼다': '자동차', '닛산': '자동차',
        '르노': '자동차', '볼보': '자동차', '재규어': '자동차', '랜드로버': '자동차',
        '전기차': '자동차', '자율주행': '자동차', '배터리': '자동차', '모빌리티': '자동차',
        
        # IT/전자
        '삼성전자': 'IT/전자', '삼성': 'IT/전자', 'LG전자': 'IT/전자', 'LG': 'IT/전자',
        '네이버': 'IT/전자', '카카오': 'IT/전자', '구글': 'IT/전자', '애플': 'IT/전자',
        'SK하이닉스': 'IT/전자', '하이닉스': 'IT/전자', '메타': 'IT/전자', '페이스북': 'IT/전자',
        '마이크로소프트': 'IT/전자', '아마존': 'IT/전자', '넷플릭스': 'IT/전자',
        'AMD': 'IT/전자', '인텔': 'IT/전자', '엔비디아': 'IT/전자', '퀄컴': 'IT/전자',
        '반도체': 'IT/전자', '인공지능': 'IT/전자', '클라우드': 'IT/전자', '디지털': 'IT/전자',
        
        # 통신
        'KT': '통신', 'SKT': '통신', 'SK텔레콤': '통신', 'LG유플러스': '통신',
        'LG U+': '통신', '티모바일': '통신', '버라이즌': '통신', 'AT&T': '통신',
        '5G': '통신', '통신망': '통신', '네트워크': '통신',
        
        # 금융
        '삼성증권': '금융', '신한은행': '금융', '국민은행': '금융', '우리은행': '금융',
        'KB': '금융', '하나은행': '금융', '삼성생명': '금융', '한화생명': '금융',
        '교보생명': '금융', '현대해상': '금융', '삼성화재': '금융', 'KB증권': '금융',
        'NH투자증권': '금융', '미래에셋': '금융', '대신증권': '금융',
        '핀테크': '금융', '가상화폐': '금융', '블록체인': '금융', '투자': '금융',
        
        # 화학/에너지
        'LG화학': '화학/에너지', '삼성화학': '화학/에너지', 'SK에너지': '화학/에너지',
        '한화': '화학/에너지', '포스코': '화학/에너지', 'GS칼텍스': '화학/에너지',
        '현대오일뱅크': '화학/에너지', 'S-Oil': '화학/에너지', '롯데케미칼': '화학/에너지',
        '한화솔루션': '화학/에너지', '한화케미칼': '화학/에너지',
        '친환경': '화학/에너지', '탄소중립': '화학/에너지', '재생에너지': '화학/에너지',
        '수소': '화학/에너지', '태양광': '화학/에너지', '풍력': '화학/에너지',
        
        # 항공/운송
        '대한항공': '항공/운송', '아시아나': '항공/운송', '진에어': '항공/운송',
        'CJ대한통운': '항공/운송', '롯데택배': '항공/운송', '한진': '항공/운송',
        '현대글로비스': '항공/운송', '팬오션': '항공/운송', 'HMM': '항공/운송',
        '물류': '항공/운송', '배송': '항공/운송', '운송': '항공/운송',
        
        # 건설/부동산
        '삼성물산': '건설/부동산', '현대건설': '건설/부동산', '대우건설': '건설/부동산',
        'GS건설': '건설/부동산', '롯데건설': '건설/부동산', '포스코건설': '건설/부동산',
        'SK건설': '건설/부동산', '한화건설': '건설/부동산', '대림건설': '건설/부동산',
        '현대엔지니어링': '건설/부동산', '삼성엔지니어링': '건설/부동산',
        '개발': '건설/부동산', '분양': '건설/부동산', '재건축': '건설/부동산',
        '리모델링': '건설/부동산', '부동산': '건설/부동산',
        
        # 유통/소비재
        '롯데': '유통/소비재', '신세계': '유통/소비재', '현대백화점': '유통/소비재',
        '이마트': '유통/소비재', '홈플러스': '유통/소비재', '코스트코': '유통/소비재',
        '쿠팡': '유통/소비재', '11번가': '유통/소비재', 'G마켓': '유통/소비재',
        '옥션': '유통/소비재', '위메프': '유통/소비재', '티몬': '유통/소비재',
        '이커머스': '유통/소비재', '온라인': '유통/소비재', '소비': '유통/소비재',
        
        # 제약/바이오
        '삼성바이오로직스': '제약/바이오', '셀트리온': '제약/바이오', '유한양행': '제약/바이오',
        '종근당': '제약/바이오', '한미약품': '제약/바이오', '대웅제약': '제약/바이오',
        '녹십자': '제약/바이오', 'SK바이오팜': '제약/바이오', '에이비엘바이오': '제약/바이오',
        '신라젠': '제약/바이오', '메디톡스': '제약/바이오', '제넥신': '제약/바이오',
        '백신': '제약/바이오', '치료제': '제약/바이오', '임상': '제약/바이오',
        '신약': '제약/바이오', '의료': '제약/바이오', '바이오': '제약/바이오',
        
        # 엔터테인먼트/미디어
        'SM': '엔터테인먼트/미디어', 'YG': '엔터테인먼트/미디어', 'JYP': '엔터테인먼트/미디어',
        'CJ': '엔터테인먼트/미디어', 'CJ ENM': '엔터테인먼트/미디어', 'JTBC': '엔터테인먼트/미디어',
        'tvN': '엔터테인먼트/미디어', 'KBS': '엔터테인먼트/미디어', 'MBC': '엔터테인먼트/미디어',
        'SBS': '엔터테인먼트/미디어', 'YTN': '엔터테인먼트/미디어',
        '하이브': '엔터테인먼트/미디어', '와이지': '엔터테인먼트/미디어',
        '콘텐츠': '엔터테인먼트/미디어', '플랫폼': '엔터테인먼트/미디어',
        '스트리밍': '엔터테인먼트/미디어', 'OTT': '엔터테인먼트/미디어',
        
        # 게임
        'NCSoft': '게임', '넥슨': '게임', '카카오게임즈': '게임', '펄어비스': '게임',
        '크래프톤': '게임', '컴투스': '게임', '웹젠': '게임', '네오위즈': '게임',
        '네트마블': '게임', '니트로': '게임', '드래곤플라이': '게임',
        '메타버스': '게임', '게임': '게임', '모바일게임': '게임', 'PC게임': '게임',
        
        # 식품/음료
        '농심': '식품/음료', '오리온': '식품/음료', '롯데제과': '식품/음료',
        '해태': '식품/음료', '빙그레': '식품/음료', '매일유업': '식품/음료',
        '남양유업': '식품/음료', '동원F&B': '식품/음료', 'CJ제일제당': '식품/음료',
        '롯데칠성': '식품/음료', '코카콜라': '식품/음료', '펩시': '식품/음료',
        '건강': '식품/음료', '프리미엄': '식품/음료', '친환경': '식품/음료',
        '유기농': '식품/음료', '가공식품': '식품/음료', '음료': '식품/음료',
        
        # 스포츠/레저
        '나이키': '스포츠/레저', '아디다스': '스포츠/레저', '푸마': '스포츠/레저',
        '스포츠': '스포츠/레저', '운동': '스포츠/레저', '헬스': '스포츠/레저',
        '레저': '스포츠/레저', '골프': '스포츠/레저', '축구': '스포츠/레저',
        '야구': '스포츠/레저', '농구': '스포츠/레저', '테니스': '스포츠/레저',
        
        # 교육
        '교육': '교육', '에듀테크': '교육', '온라인교육': '교육', '학원': '교육',
        '대학': '교육', '학교': '교육', '수업': '교육', '강의': '교육',
        
        # 정부/공공기관
        '정부': '정부/공공', '국정원': '정부/공공', '경찰': '정부/공공',
        '소방서': '정부/공공', '공공기관': '정부/공공', '지자체': '정부/공공',
        '시청': '정부/공공', '구청': '정부/공공', '도청': '정부/공공',
        
        # 기타
        '기타': '기타', '일반': '기타', '사회': '기타', '문화': '기타',
        '종교': '기타', '환경': '기타', '사건': '기타', '사고': '기타'
    }
    
    print(f"✅ 총 {len(company_to_industry)}개 기업-업계 매핑 로드 완료")
    return company_to_industry

# ──────────── 기업명 추출: soynlp → konlpy Okt ────────────

def extract_company_names_okt(title, content=""):
    """konlpy Okt를 사용한 기업명 추출"""
    try:
        full_text = f"{title} {content}".strip()
        if not full_text:
            return []
        
        # 명사 추출
        nouns = okt.nouns(full_text)
        
        # 기업명 후보 추출
        companies = set()
        
        # 기업명 패턴 매칭
        company_patterns = [
            r'([가-힣]{2,8}(?:전자|자동차|건설|화학|제약|통신|금융|보험|은행|카드|증권|투자|그룹|홀딩스|바이오|게임|엔터|미디어))',
            r'([가-힣]{2,6}(?:주식회사|㈜|회사|기업|산업|물산|상사|개발|테크|소프트|시스템))',
            r'(현대[가-힣]{0,4}|삼성[가-힣]{0,4}|LG[가-힣]{0,4}|SK[가-힣]{0,4}|롯데[가-힣]{0,4})',
            r'(GS[가-힣]{0,4}|CJ[가-힣]{0,4}|한화[가-힣]{0,4}|포스코[가-힣]{0,4})',
            r'([A-Z]{2,6}[가-힣]{0,4})',  # 영문+한글 조합
            r'([가-힣]{2,6}[A-Z]{1,4})',  # 한글+영문 조합
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, full_text)
            companies.update(matches)
        
        # 명사에서 기업명 찾기
        for noun in nouns:
            if (len(noun) >= 2 and len(noun) <= 8 and
                re.search(r'[가-힣]', noun) and
                not noun.isdigit()):
                companies.add(noun)
        
        # 불용어 제거
        stopwords = {
            '기자', '사진', '제공', '관련', '업계', '시장', '산업', '분야', '회사', '기업',
            '발표', '공개', '설명', '말했다', '밝혔다', '전했다', '보도', '뉴스', '기사',
            '대표', '사장', '회장', '부사장', '이사', '상무', '전무', '본부장', '팀장',
            '오늘', '어제', '내일', '최근', '현재', '당시', '이번', '다음', '지난'
        }
        
        # 최종 기업명 필터링
        final_companies = []
        for company in companies:
            if (company not in stopwords and
                len(company) >= 2 and len(company) <= 10 and
                not company.isdigit() and
                re.search(r'[가-힣]', company)):
                final_companies.append(company)
        
        return final_companies
        
    except Exception as e:
        print(f"konlpy 기업명 추출 실패: {e}")
        return extract_company_names_simple(title, content)

def extract_company_names_simple(title, content=""):
    """간단한 기업명 추출 (백업용)"""
    full_text = f"{title} {content}".strip()
    if not full_text:
        return []
    
    # 기업명 패턴들
    company_patterns = [
        r'([가-힣]{2,8}(?:전자|자동차|건설|화학|제약|통신|금융|보험|은행|카드|증권|투자|그룹|홀딩스|바이오|게임|엔터|미디어))',
        r'([가-힣]{2,6}(?:주식회사|㈜|회사|기업|산업|물산|상사|개발|테크|소프트|시스템))',
        r'(현대[가-힣]{0,4}|삼성[가-힣]{0,4}|LG[가-힣]{0,4}|SK[가-힣]{0,4}|롯데[가-힣]{0,4})',
        r'(GS[가-힣]{0,4}|CJ[가-힣]{0,4}|한화[가-힣]{0,4}|포스코[가-힣]{0,4})',
        r'([A-Z]{2,6}[가-힣]{0,4})',  # 영문+한글 조합
        r'([가-힣]{2,6}[A-Z]{1,4})',  # 한글+영문 조합
    ]
    
    companies = set()
    
    # 패턴 매칭
    for pattern in company_patterns:
        matches = re.findall(pattern, full_text)
        companies.update(matches)
    
    # 단어 기반 필터링
    words = full_text.split()
    stopwords = {
        '기자', '사진', '제공', '관련', '업계', '시장', '산업', '분야', '회사', '기업',
        '발표', '공개', '설명', '말했다', '밝혔다', '전했다', '보도', '뉴스', '기사',
        '대표', '사장', '회장', '부사장', '이사', '상무', '전무', '본부장', '팀장'
    }
    
    for word in words:
        word = word.strip()
        if (len(word) >= 2 and len(word) <= 10 and
            word not in stopwords and
            not word.isdigit() and
            re.search(r'[가-힣]', word)):
            companies.add(word)
    
    return list(companies)

def map_industry_advanced(row, company_to_industry):
    """고급 업계 매핑 (konlpy 활용)"""
    # 제목 찾기
    title_columns = ['제목', 'title', '헤드라인', '기사제목', '기사 제목']
    title = ""
    for col in title_columns:
        if col in row and pd.notna(row[col]):
            title = str(row[col]).strip()
            break
    
    # 내용 찾기
    content_columns = ['내용', 'content', '본문', '기사내용', '주요내용']
    content = ""
    for col in content_columns:
        if col in row and pd.notna(row[col]):
            content = str(row[col]).strip()
            break
    
    if not title:
        return '기타'
    
    # konlpy 기반 기업명 추출
    companies = extract_company_names_okt(title, content)
    
    if not companies:
        return '기타'
    
    # 업계 매핑
    matched_industries = []
    for company in companies:
        # 정확한 매칭
        if company in company_to_industry:
            matched_industries.append(company_to_industry[company])
            continue
        
        # 부분 매칭
        for mapped_company, industry in company_to_industry.items():
            if company in mapped_company or mapped_company in company:
                matched_industries.append(industry)
                break
    
    if matched_industries:
        # 가장 많이 매칭된 업계 반환
        industry_counts = Counter(matched_industries)
        return industry_counts.most_common(1)[0][0]
    
    return '기타'

# ──────────── 업계별 키워드 추출: soynlp → konlpy Okt ────────────

def extract_industry_keywords(df, industry):
    """업계별 특화 키워드 추출 (konlpy 사용)"""
    if industry != '전체':
        industry_articles = df[df['업계'] == industry]
    else:
        industry_articles = df
    
    titles = industry_articles['제목'].dropna().tolist()
    
    # konlpy Okt로 업계 특화 키워드 추출
    if not titles:
        return []
    
    # 전체 텍스트 결합
    all_text = " ".join(titles)
    
    # 명사 추출
    nouns = okt.nouns(all_text)
    freq = Counter(nouns)
    
    # 불용어 제거
    stopwords = {
        '기자', '사진', '제공', '관련', '업계', '시장', '분야', '회사', '기업',
        '발표', '공개', '설명', '보도', '뉴스', '기사', '오늘', '어제', '내일'
    }
    
    keywords = [(w, freq[w]) for w in freq if len(w) > 1 and w not in stopwords]
    keywords.sort(key=lambda x: x[1], reverse=True)
    
    # 업계별 가중치 조정
    industry_weights = {
        '자동차': ['전기차', '자율주행', '배터리', '모빌리티', '전동화'],
        'IT/전자': ['AI', '반도체', '클라우드', '디지털', '인공지능'],
        '금융': ['핀테크', '가상화폐', '블록체인', '투자', '디지털뱅킹'],
        '바이오': ['백신', '치료제', '임상', '신약', '의료'],
        '화학/에너지': ['친환경', '탄소중립', '재생에너지', '수소'],
        '항공/운송': ['물류', '배송', '모빌리티', '운송'],
        '건설/부동산': ['개발', '분양', '재건축', '리모델링'],
        '유통/소비재': ['이커머스', '온라인', '배송', '소비'],
        '엔터테인먼트/미디어': ['콘텐츠', '플랫폼', '스트리밍', 'OTT'],
        '게임': ['메타버스', '게임', '플랫폼', '모바일'],
        '식품/음료': ['건강', '프리미엄', '친환경', '유기농']
    }
    
    if industry in industry_weights:
        boost_keywords = industry_weights[industry]
        enhanced_keywords = []
        for word, score in keywords:
            if any(boost in word for boost in boost_keywords):
                enhanced_keywords.append((word, score * 1.5))
            else:
                enhanced_keywords.append((word, score))
        return sorted(enhanced_keywords, key=lambda x: x[1], reverse=True)[:20]
    
    return keywords[:20]

def add_industry_column(df):
    """DataFrame에 업계 컬럼 추가"""
    if df is None or df.empty:
        return df
    
    print("🚀 업계 매핑 시작...")
    
    # 업계 매핑 데이터 로드
    company_to_industry = load_industry_mapping_advanced()
    
    if not company_to_industry:
        print("❌ 업계 매핑 데이터가 없습니다.")
        return df
    
    # 업계 컬럼 초기화
    if '업계' not in df.columns:
        df['업계'] = '기타'
    
    total_rows = len(df)
    mapped_count = 0
    
    # 각 행에 대해 업계 매핑
    for idx, row in df.iterrows():
        if pd.isna(df.loc[idx, '업계']) or df.loc[idx, '업계'] == '기타':
            industry = map_industry_advanced(row, company_to_industry)
            df.loc[idx, '업계'] = industry
            if industry != '기타':
                mapped_count += 1
    
    # 결과 출력
    industry_counts = df['업계'].value_counts()
    print(f"✅ 업계 매핑 완료!")
    print(f"📊 총 {mapped_count}/{total_rows}개 기사 매핑 성공")
    print(f"📋 업계별 분포:")
    for industry, count in industry_counts.head(10).items():
        print(f"  - {industry}: {count}개")
    
    return df

# ================================
# 날짜 처리 함수들
# ================================

def extract_month_from_date(date_str):
    """날짜에서 월 추출"""
    if pd.isnull(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # 1월1일 형식 처리
    month_day_pattern = r'(\d{1,2})월(\d{1,2})일'
    match = re.search(month_day_pattern, date_str)
    
    if match:
        month = int(match.group(1))
        return f"{month}월"
    
    return None

def extract_week_from_date(date_str):
    """날짜에서 주차 추출"""
    if pd.isnull(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # 1월1일 형식 처리
    month_day_pattern = r'(\d{1,2})월(\d{1,2})일'
    match = re.search(month_day_pattern, date_str)
    
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        
        # 주차 계산
        if day <= 7:
            week = 1
        elif day <= 14:
            week = 2
        elif day <= 21:
            week = 3
        elif day <= 28:
            week = 4
        else:
            week = 5
        
        return f"{month}월{week}주차"
    
    return None

# ================================
# 네이버 링크 하이퍼링크 처리 함수들
# ================================

def get_naver_link_column(df):
    """구글 시트에서 네이버링크 컬럼명을 찾는 함수"""
    possible_columns = ['네이버링크', '네이버 링크', '네이버URL', '네이버 URL', 'naver_link', 'naver_url']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '네이버' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def get_media_link_column(df):
    """구글 시트에서 매체링크 컬럼명을 찾는 함수"""
    possible_columns = ['매체링크', '매체 링크', '매체URL', '매체 URL', 'media_link', 'media_url']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '매체' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def create_naver_hyperlink(title, naver_url=None, other_url=None, media_url=None):
    """네이버 링크를 하이퍼링크로 생성하는 함수 (매체링크 지원 추가)"""
    if pd.isna(title):
        title = "제목 없음"
    else:
        title = str(title).strip()
    
    # 1순위: 네이버 링크가 있으면 우선 사용
    if pd.notna(naver_url) and str(naver_url).strip() and str(naver_url).strip().lower() != 'nan':
        naver_url = str(naver_url).strip()
        # URL이 http로 시작하지 않으면 추가
        if not naver_url.startswith(('http://', 'https://')):
            naver_url = 'https://' + naver_url
        return f'📰 [{title}]({naver_url})'
    
    # 2순위: 매체링크가 있으면 사용
    if pd.notna(media_url) and str(media_url).strip() and str(media_url).strip().lower() != 'nan':
        media_url = str(media_url).strip()
        # URL이 http로 시작하지 않으면 추가
        if not media_url.startswith(('http://', 'https://')):
            media_url = 'https://' + media_url
        return f'📰 [{title}]({media_url})'
    
    # 3순위: 네이버 링크가 없으면 다른 URL 사용
    if pd.notna(other_url) and str(other_url).strip() and str(other_url).strip().lower() != 'nan':
        other_url = str(other_url).strip()
        # URL이 http로 시작하지 않으면 추가
        if not other_url.startswith(('http://', 'https://')):
            other_url = 'https://' + other_url
        return f'🔗 [{title}]({other_url})'
    
    return f"📄 {title}"

def validate_url(url):
    """URL 유효성 검사"""
    if pd.isna(url):
        return False
    
    url = str(url).strip()
    
    # 빈 문자열이나 'nan' 체크
    if not url or url.lower() == 'nan':
        return False
    
    # 기본 URL 패턴 체크
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']*\.com[^\s<>"\']*'
    return bool(re.match(url_pattern, url, re.IGNORECASE))

def format_article_link(article_row):
    """기사 행에서 링크 정보를 포맷팅 (매체링크 지원 추가)"""
    title = article_row.get('제목', '제목 없음')
    naver_url = article_row.get('네이버 URL') or article_row.get('네이버링크')
    media_url = article_row.get('매체 URL') or article_row.get('매체링크')
    other_url = article_row.get('기타 URL')
    
    # 1순위: 네이버 URL 확인
    if validate_url(naver_url):
        link_url = str(naver_url).strip()
        if not link_url.startswith(('http://', 'https://')):
            link_url = 'https://' + link_url
        return {
            'title': title,
            'url': link_url,
            'source': 'naver',
            'html': f'📰 [{title}]({link_url})'
        }
    
    # 2순위: 매체 URL 확인
    if validate_url(media_url):
        link_url = str(media_url).strip()
        if not link_url.startswith(('http://', 'https://')):
            link_url = 'https://' + link_url
        return {
            'title': title,
            'url': link_url,
            'source': 'media',
            'html': f'📰 [{title}]({link_url})'
        }
    
    # 3순위: 기타 URL 확인
    if validate_url(other_url):
        link_url = str(other_url).strip()
        if not link_url.startswith(('http://', 'https://')):
            link_url = 'https://' + link_url
        return {
            'title': title,
            'url': link_url,
            'source': 'other',
            'html': f'🔗 [{title}]({link_url})'
        }
    
    # 링크가 없는 경우
    return {
        'title': title,
        'url': None,
        'source': 'none',
        'html': f"📄 {title}"
    }

# ================================
# 새로운 AI 기능 함수들
# ================================

def generate_today_planning_items(df, industry="전체"):
    """상위 가중치 기사들을 바탕으로 Perplexity API를 호출하여 '오늘의 기획 아이템' 3개를 제안"""
    if df.empty:
        return "분석할 데이터가 없습니다."
    
    # 업계 필터링
    if industry != "전체":
        df = df[df["업계"] == industry]
    
    # 가중치 상위 기사 20개까지 대상
    top_articles = (
        df.nlargest(20, "전체가중치") if "전체가중치" in df.columns else df.head(20)
    )
    
    # 기사 텍스트 추출
    texts = []
    for _, article in top_articles.iterrows():
        title = article.get("제목", "")
        body = article.get("주요내용", "")
        if pd.notna(title) and pd.notna(body):
            texts.append(f"제목: {title}\n내용: {body}")
        elif pd.notna(title):
            texts.append(f"제목: {title}")
    
    if not texts:
        return "분석할 기사 내용이 없습니다."
    
    # Perplexity API 호출용 프롬프트 생성
    prompt = f"""
    다음은 오늘의 주요 뉴스 기사들입니다. 이 기사들을 바탕으로 기자들이 추가 취재하기 좋은 '기획 아이템' 3건을 제안해주세요.
    
    기사 목록:
    {chr(10).join(texts[:15])}
    
    [답변 형식]
    1. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    2. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    3. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    """
    
    # API 호출 (캐시 활용)
    try:
        from app import call_perplexity_api_cached
        result = call_perplexity_api_cached(prompt, max_age_hours=6)
        return result or "기획 아이템 생성에 실패했습니다."
    except ImportError:
        return "AI 기능을 사용할 수 없습니다. app.py 모듈을 확인해주세요."

def generate_monthly_weekly_insights(df, target_month=None):
    """월별·주차별 핵심 문장 생성 (Perplexity API 활용)"""
    if df.empty:
        return {}
    
    # 월별 필터링
    if target_month and "월" in df.columns:
        df = df[df["월"] == target_month]
    
    if df.empty:
        return {}
    
    # 월 전체 핵심 문장 생성
    month_texts = []
    for _, row in df.head(50).iterrows():
        title = row.get("제목", "")
        content = row.get("주요내용", "")
        if pd.notna(title):
            month_texts.append(f"{title} {content if pd.notna(content) else ''}")
    
    monthly_prompt = f"""
    다음은 {target_month or '이번 달'} 주요 뉴스들입니다.
    이 뉴스들의 핵심 이슈를 한 문장으로 요약해 주세요.
    
    뉴스 내용:
    {' '.join(month_texts)}
    """
    
    # 주차별 핵심 문장 생성
    weekly_insights = {}
    if "주차" in df.columns:
        for week in sorted(df["주차"].dropna().unique()):
            week_df = df[df["주차"] == week]
            week_texts = []
            for _, row in week_df.head(20).iterrows():
                title = row.get("제목", "")
                content = row.get("주요내용", "")
                if pd.notna(title):
                    week_texts.append(f"{title} {content if pd.notna(content) else ''}")
            
            weekly_prompt = f"""
            다음은 {week} 주요 뉴스들입니다.
            이 뉴스들의 핵심 이슈를 한 문장으로 요약해 주세요.
            
            뉴스 내용:
            {' '.join(week_texts)}
            """
            
            # API 호출
            try:
                from app import call_perplexity_api_cached
                weekly_insights[week] = call_perplexity_api_cached(
                    weekly_prompt, max_age_hours=24
                )
            except ImportError:
                weekly_insights[week] = "AI 기능을 사용할 수 없습니다."
    
    # 월별 핵심 문장 API 호출
    try:
        from app import call_perplexity_api_cached
        monthly_insight = call_perplexity_api_cached(monthly_prompt, max_age_hours=12)
    except ImportError:
        monthly_insight = "AI 기능을 사용할 수 없습니다."
    
    return {
        "monthly_insight": monthly_insight,
        "weekly_insights": weekly_insights
    }

def find_similar_articles_to_insight(df, insight, top_n=5):
    """핵심 문장과 유사도가 높은 기사 n개 반환"""
    if df.empty or not insight:
        return []
    
    # 기사 텍스트 준비
    df = df.copy()
    df["combined_text"] = df.apply(
        lambda row: f"{row.get('제목', '')} {row.get('주요내용', '')}", axis=1
    )
    
    df = df[df["combined_text"].str.strip() != ""]
    
    if df.empty:
        return []
    
    # TF-IDF 유사도 계산
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(max_features=1000)
        corpus = df["combined_text"].tolist() + [insight]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # 유사도 계산 (마지막 요소가 insight)
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
        
        # 상위 유사도 기사 선택
        similar_indices = similarities.argsort()[::-1][:top_n]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # 최소 유사도 임계값
                results.append({
                    "article": df.iloc[idx],
                    "similarity": similarities[idx]
                })
        
        return results
        
    except ImportError:
        return []

# ================================
# 메인 전처리 함수
# ================================

def preprocess_dataframe(df):
    """DataFrame 전처리"""
    if df is None or df.empty:
        return df
    
    try:
        print("🚀 데이터 전처리 시작...")
        
        # 기본 전처리
        df = df.dropna(how='all')
        df.columns = df.columns.str.strip()
        df = df.replace('', pd.NA)
        
        # 날짜 처리
        if '보도날짜' in df.columns:
            if '월' not in df.columns:
                df['월'] = df['보도날짜'].apply(extract_month_from_date)
            if '주차' not in df.columns:
                df['주차'] = df['보도날짜'].apply(extract_week_from_date)
        
        # 업계 컬럼 자동 추가
        df = add_industry_column(df)
        
        # 가중치 계산
        if '구분(일반/단독)' in df.columns:
            df['가중치'] = df.apply(calculate_article_weight, axis=1)
        
        # 시간 가중치 계산
        if '보도날짜' in df.columns:
            df['시간가중치'] = df['보도날짜'].apply(calculate_time_weight)
        
        # 전체 가중치 계산 (기존 가중치 + 시간 가중치)
        if '가중치' in df.columns:
            df['전체가중치'] = df['가중치'] * 0.7 + df['시간가중치'] * 0.3
        else:
            df['전체가중치'] = df['시간가중치']
        
        # 네이버 링크 처리 컬럼 추가
        if '제목' in df.columns:
            df['링크정보'] = df.apply(lambda row: format_article_link(row), axis=1)
        
        print("✅ 데이터 전처리 완료!")
        print(f"📊 최종 데이터: {len(df)}행, {len(df.columns)}열")
        
        return df
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return df

# ================================
# 최종 초기화
# ================================

print("✅ Google Sheets 유틸리티 모듈 로드 완료")
print("📊 konlpy 기반 한국어 처리 엔진 사용")
print("🚀 구글 시트 데이터 캐싱 시스템 활성화")

# 버전 정보
__version__ = "3.3.0-konlpy-cached"
__author__ = "뉴스팩토리 개발팀"
__description__ = "konlpy 기반 한국어 뉴스 분석 + 구글 시트 캐싱 시스템 + 링크 하이퍼링크"

print(f"🚀 모듈 버전: {__version__}")
print(f"👥 개발자: {__author__}")
print(f"📝 설명: {__description__}")
