from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from services.weather_service import Weather
from services.soil_service import Soil

@dataclass
class TreeParams:
    species: str               # "Береза", "Дуб", "Ель", "Сосна", "Тополь"
    height_m: float
    crown_width_m: float
    trunk_diameter_m: float    # на 1.3 м, как у тебя

@dataclass
class RiskResult:
    index: float          # 0..1
    category: str         # "низкий", "средний", "высокий"
    explanation: list[str]

SPECIES_BASE = {
    "Береза": 0.7,
    "Дуб": 0.5,
    "Ель": 1.0,
    "Сосна": 0.75,
    "Тополь": 0.95,
}

def _slenderness_score(S: float) -> float:
    # H/D
    if S >= 80:
        return 1.0
    if S >= 60:
        return 0.7
    if S >= 40:
        return 0.4
    return 0.2

def _soil_score(soil: Optional[Soil]) -> float:
    if soil is None:
        return 0.5  # нейтрально, если данных нет

    clay = soil.clay or 0
    sand = soil.sand or 0
    org = soil.organic_carbon or 0

    # очень простая схема:
    if org > 80:              # органические/торфяные почвы
        return 1.0
    if clay > 40:             # тяжелые глины, часто переувлажнены
        return 0.9
    if sand > 60:             # лёгкие пески
        return 0.7
    return 0.5                # суглинки, условная норма

def _wind_score(weather: Optional[Weather]) -> float:
    if weather is None:
        return 0.5
    gust = weather.wind_gust or weather.wind_speed or 0.0
    if gust <= 5:
        return 0.2
    if gust <= 10:
        return 0.4
    if gust <= 15:
        return 0.6
    if gust <= 25:
        return 0.8
    return 1.0  # >25 м/с – уже штормовой ветер

def compute_risk(tree: TreeParams,
                 weather: Optional[Weather],
                 soil: Optional[Soil]) -> RiskResult:
    expl: list[str] = []

    # 1. Порода
    base = SPECIES_BASE.get(tree.species, 0.7)
    expl.append(f"Порода дерева: {tree.species}, базовый коэффициент риска {base:.2f}.")

    # 2. Стройность
    if tree.trunk_diameter_m <= 0:
        S = 80  # худший случай
    else:
        S = tree.height_m / tree.trunk_diameter_m
    s_score = _slenderness_score(S)
    expl.append(f"Коэффициент стройности H/D = {S:.1f}, вклад в риск {s_score:.2f}.")

    # 3. Ветер
    w_score = _wind_score(weather)
    if weather is not None and weather.wind_speed is not None:
        expl.append(
            f"Текущая скорость ветра {weather.wind_speed:.1f} м/с, "
            f"порывы до { (weather.wind_gust or weather.wind_speed):.1f} м/с, "
            f"ветровой фактор {w_score:.2f}."
        )
    else:
        expl.append("Нет данных о ветре, ветровой фактор принят усреднённым 0.5.")

    # 4. Почва
    soil_score = _soil_score(soil)
    if soil is None:
        expl.append("Данные о почве недоступны, почвенный фактор принят 0.5.")
    else:
        expl.append(
            f"Почва: глина ≈ {soil.clay or 0:.0f} %, песок ≈ {soil.sand or 0:.0f} %, "
            f"органика ≈ {soil.organic_carbon or 0:.0f}, почвенный фактор {soil_score:.2f}."
        )

    # Сводим к единому индексу (веса можно потом подправить)
    risk_raw = (
        0.3 * base +
        0.3 * s_score +
        0.25 * w_score +
        0.15 * soil_score
    )

    # Нормируем в 0..1
    risk_index = max(0.0, min(risk_raw, 1.0))

    if risk_index < 0.4:
        category = "низкий"
    elif risk_index < 0.7:
        category = "средний"
    else:
        category = "высокий"

    expl.append(f"Итоговый индекс риска: {risk_index:.2f} ({category}).")

    return RiskResult(
        index=risk_index,
        category=category,
        explanation=expl,
    )
