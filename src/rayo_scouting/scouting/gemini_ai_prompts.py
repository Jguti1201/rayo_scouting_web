"""
ai_prompts.py — Prompts IA (v2)
===============================
Centraliza prompts usados por la app de scouting.

MEJORAS v2
----------
1. Tono más institucional / dirección deportiva.
2. Prompt de similitud más orientado a decisión.
3. Añade prompt breve para resumen ejecutivo.
"""

from __future__ import annotations
from typing import Dict, Any

def generate_player_similarity_prompt(
    target_player: Dict[str, Any],
    candidate_player: Dict[str, Any],
    similarity_score: float,
    adaptation_score: float
) -> str:
    return f"""
Eres un analista profesional de scouting del Rayo Vallecano.
Debes redactar una valoración técnica y ejecutiva sobre por qué un jugador candidato
puede ser una alternativa válida al jugador objetivo, combinando similitud estadística
y adaptación al contexto del club.

JUGADOR OBJETIVO
- Nombre: {target_player.get('Name', 'N/A')}
- Posición: {target_player.get('posicion', 'N/A')}
- Club: {target_player.get('tm_club', 'N/A')}
- Liga: {target_player.get('liga', 'N/A')}
- Edad: {target_player.get('edad', 'N/A')}
- Rol táctico: {target_player.get('cluster_label', 'N/A')}

JUGADOR CANDIDATO
- Nombre: {candidate_player.get('Name', 'N/A')}
- Posición: {candidate_player.get('posicion', 'N/A')}
- Club: {candidate_player.get('tm_club', 'N/A')}
- Liga: {candidate_player.get('liga', 'N/A')}
- Edad: {candidate_player.get('edad', 'N/A')}
- Rol táctico: {candidate_player.get('cluster_label', 'N/A')}
- Valor de mercado: {candidate_player.get('valor_mercado', 'N/A')}
- Fin de contrato: {candidate_player.get('fin_contrato', 'N/A')}

INDICADORES
- Similitud estadística: {similarity_score:.1f}%
- Adaptación al Rayo: {adaptation_score:.1f}/100

CONTEXTO DE EQUIPO
- Club objetivo: Rayo Vallecano
- Filosofía: intensidad competitiva, juego vertical, transiciones y presión
- Necesidad: encontrar perfiles funcionales y sostenibles para plantilla competitiva

INSTRUCCIONES
1. Explica por qué el candidato se parece al jugador objetivo.
2. Describe en qué aspectos podría encajar bien en el Rayo.
3. Señala fortalezas concretas.
4. Indica riesgos o puntos de atención.
5. Cierra con una valoración ejecutiva breve.
6. Responde en español.
7. Extensión: 220-320 palabras.
"""

def generate_comprehensive_report_prompt(
    target_player: Dict[str, Any],
    candidates: list,
    top_n: int = 5
) -> str:
    candidates_text = "\n".join([
        f"- {c.get('Name', 'N/A')} ({c.get('tm_club', 'N/A')}, {c.get('liga', 'N/A')}): similitud {c.get('similarity', 0)*100:.1f}%, adaptación {c.get('adapt_score', 0):.1f}"
        for c in candidates[:top_n]
    ])

    return f"""
Eres un director deportivo que debe resumir una shortlist de scouting para el Rayo Vallecano.

JUGADOR DE REFERENCIA
- Nombre: {target_player.get('Name', 'N/A')}
- Posición: {target_player.get('posicion', 'N/A')}

TOP {top_n} CANDIDATOS
{candidates_text}

ESTRUCTURA DEL INFORME
1. Resumen ejecutivo
2. Lectura comparativa de los perfiles
3. Recomendación priorizada
4. Riesgos y condicionantes
5. Conclusión final

Responde en español con tono profesional y claro.
"""

def generate_executive_summary_prompt(
    squad_need: str,
    candidate_name: str,
    candidate_context: Dict[str, Any]
) -> str:
    return f"""
Redacta un resumen ejecutivo muy breve para dirección deportiva del Rayo Vallecano.

NECESIDAD DETECTADA:
{squad_need}

CANDIDATO:
- Nombre: {candidate_name}
- Club: {candidate_context.get('tm_club', 'N/A')}
- Liga: {candidate_context.get('liga', 'N/A')}
- Edad: {candidate_context.get('edad', 'N/A')}
- Valor de mercado: {candidate_context.get('valor_mercado', 'N/A')}
- Posición: {candidate_context.get('posicion', 'N/A')}

Devuelve:
- Encaje principal
- Principal fortaleza
- Principal riesgo
- Recomendación final

Máximo 120 palabras.
"""