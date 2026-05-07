import json
import os
import anthropic


def generate_report(image_base64: str, fractura: bool, confianza: float) -> dict:
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    resultado = "FRACTURA DETECTADA" if fractura else "SIN FRACTURA DETECTADA"

    prompt = (
        f"Eres un sistema de apoyo radiologico por IA. Analiza esta imagen de rayos X.\n\n"
        f"El modelo de clasificacion CNN (ResNet50) ha determinado:\n"
        f"- Resultado: {resultado}\n"
        f"- Confianza: {confianza:.1f}%\n\n"
        f"Genera un informe radiologico orientativo en formato JSON con esta estructura exacta:\n"
        f'{{\n'
        f'  "region": "zona anatomica evaluada (ej: miembro superior, cintura escapular, rodilla, etc.)",\n'
        f'  "hueso": "hueso posiblemente afectado (ej: clavicula, femur, radio, etc.)",\n'
        f'  "hallazgos": ["hallazgo 1", "hallazgo 2", "hallazgo 3"],\n'
        f'  "estructuras": ["estructura evaluada 1", "estructura evaluada 2"],\n'
        f'  "conclusion": "conclusion orientativa del analisis",\n'
        f'  "manejo": "posible manejo medico sugerido segun los hallazgos",\n'
        f'  "recuperacion": "tiempo aproximado de recuperacion con y sin cirugia si aplica",\n'
        f'  "recomendaciones": ["recomendacion 1", "recomendacion 2", "recomendacion 3"],\n'
        f'  "alertas": ["signo de alarma 1", "signo de alarma 2", "signo de alarma 3"]\n'
        f'}}\n\n'
        f"INSTRUCCIONES:\n"
        f"- Basa tu analisis en lo que OBSERVAS en la imagen radiografica\n"
        f"- Se especifico sobre la zona anatomica, tipo de hueso y localizacion\n"
        f"- Si hay fractura, describe el tipo probable (transversal, oblicua, espiral, conminuta, etc.)\n"
        f"- Si hay fractura, indica la ubicacion especifica (tercio proximal, medio, distal, etc.)\n"
        f"- Incluye informacion sobre desplazamiento de fragmentos si se observa\n"
        f"- Si NO hay fractura, describe las estructuras normales observadas\n"
        f"- Usa lenguaje medico pero comprensible\n"
        f"- Todo es ORIENTATIVO, no diagnostico definitivo\n"
        f"- Responde SOLO con el JSON valido, sin texto adicional ni bloques de codigo"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    text = response.content[0].text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        if '```' in text:
            text = text.rsplit('```', 1)[0]
        text = text.strip()

    return json.loads(text)
