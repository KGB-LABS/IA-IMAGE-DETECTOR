import gradio as gr
import numpy as np
import cv2
from PIL import Image
import io
from scipy import signal
from scipy.fftpack import fft2, fftshift
from skimage import exposure
import warnings
warnings.filterwarnings('ignore')


class AIImageDetector:
    """Classe principal para análise forense de imagens"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_image(self, image_input):
       
        if isinstance(image_input, str):
            img_pil = Image.open(image_input)
        else:
            img_pil = image_input
        
      
        img_array = np.array(img_pil)
        
       
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array
        
       
        results = {
            'error_level_analysis': self._error_level_analysis(img_pil),
            'noise_analysis': self._noise_analysis(img_cv),
            'frequency_analysis': self._frequency_analysis(img_cv),
            'compression_analysis': self._compression_analysis(img_pil),
            'metadata_analysis': self._metadata_analysis(img_pil),
        }
        
        
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results, img_pil
    
    def _error_level_analysis(self, img_pil):
       
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        img_recompressed = Image.open(buffer)
        
        
        img_array = np.array(img_pil, dtype=np.float32)
        recomp_array = np.array(img_recompressed, dtype=np.float32)
        
        
        if img_array.shape != recomp_array.shape:
            recomp_array = cv2.resize(recomp_array, (img_array.shape[1], img_array.shape[0]))
        
        diff = np.abs(img_array - recomp_array)
        
        
        ela_mean = np.mean(diff)
        ela_std = np.std(diff)
        ela_max = np.percentile(diff, 95)
        
        
        ela_uniformity = 1.0 - min(ela_std / (ela_mean + 1e-6), 1.0)
        
        return {
            'mean_error': float(ela_mean),
            'std_error': float(ela_std),
            'max_error': float(ela_max),
            'uniformity_score': float(ela_uniformity),
            'suspicious': ela_uniformity > 0.7
        }
    
    def _noise_analysis(self, img_cv):
        """
       
        """
        if len(img_cv.shape) == 3:
            # Converter para escala de cinza
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
        
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        
        h, w = gray.shape
        regions = []
        region_size = 64
        
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                variance = np.var(region)
                regions.append(variance)
        
        if regions:
            noise_consistency = 1.0 - (np.std(regions) / (np.mean(regions) + 1e-6))
            noise_consistency = min(max(noise_consistency, 0), 1.0)
        else:
            noise_consistency = 0.5
        
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'noise_consistency': float(noise_consistency),
            'entropy': float(entropy),
            'laplacian_variance': float(np.var(laplacian)),
            'suspicious': noise_consistency > 0.75
        }
    
    def _frequency_analysis(self, img_cv):
       
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
        
        
        gray_resized = cv2.resize(gray, (256, 256))
        
        
        f_transform = fft2(gray_resized)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        
        center = magnitude.shape[0] // 2
        
        
        low_freq = magnitude[center-30:center+30, center-30:center+30]
        low_energy = np.sum(low_freq ** 2)
        
        
        high_freq = np.concatenate([
            magnitude[:20, :].flatten(),
            magnitude[-20:, :].flatten(),
            magnitude[:, :20].flatten(),
            magnitude[:, -20:].flatten()
        ])
        high_energy = np.sum(high_freq ** 2)
        
        
        freq_ratio = low_energy / (high_energy + 1e-6)
        
        
        freq_suspicious = freq_ratio > 100 or freq_ratio < 0.01
        
        return {
            'low_frequency_energy': float(low_energy),
            'high_frequency_energy': float(high_energy),
            'frequency_ratio': float(freq_ratio),
            'suspicious': freq_suspicious
        }
    
    def _compression_analysis(self, img_pil):
        
        is_jpeg = img_pil.format == 'JPEG'
        
        if is_jpeg:
           
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            img_recompressed = Image.open(buffer)
            
            img_array = np.array(img_pil, dtype=np.float32)
            recomp_array = np.array(img_recompressed, dtype=np.float32)
            
            if img_array.shape == recomp_array.shape:
                compression_artifacts = np.mean(np.abs(img_array - recomp_array))
            else:
                compression_artifacts = 0
        else:
            compression_artifacts = 0
        
        return {
            'is_jpeg': is_jpeg,
            'compression_artifacts': float(compression_artifacts),
            'suspicious': compression_artifacts > 10 if is_jpeg else False
        }
    
    def _metadata_analysis(self, img_pil):
       
        metadata = {
            'format': img_pil.format,
            'mode': img_pil.mode,
            'size': img_pil.size,
            'has_exif': False,
            'suspicious_metadata': False
        }
        
      
        try:
            exif_data = img_pil._getexif()
            metadata['has_exif'] = exif_data is not None
        except:
            metadata['has_exif'] = False
        
        
        metadata['suspicious_metadata'] = not metadata['has_exif']
        
        return metadata
    
    def _calculate_overall_score(self, results):
        
        scores = []
        
        
        ela = results['error_level_analysis']
        scores.append(ela['uniformity_score'] * 100)
        
        
        noise = results['noise_analysis']
        scores.append(noise['noise_consistency'] * 100)
        
        
        freq = results['frequency_analysis']
        if freq['suspicious']:
            scores.append(75)
        else:
            scores.append(25)
        
        #
        comp = results['compression_analysis']
        if comp['suspicious']:
            scores.append(60)
        else:
            scores.append(20)
        
        
        meta = results['metadata_analysis']
        if meta['suspicious_metadata']:
            scores.append(40)
        else:
            scores.append(10)
        
        overall = np.mean(scores)
        
        return {
            'probability_ai': float(overall),
            'confidence': 'Alta' if overall > 70 else 'Média' if overall > 40 else 'Baixa',
            'verdict': 'Provável IA' if overall > 70 else 'Incerto' if overall > 40 else 'Provável Real'
        }


def format_results(results, img_pil):
   
    
    output_text = "=" * 60 + "\n"
    output_text += "RELATÓRIO DE ANÁLISE DE IMAGEM\n"
    output_text += "=" * 60 + "\n\n"
    

    overall = results['overall_score']
    output_text += f"🔍 VEREDICTO GERAL\n"
    output_text += f"   Probabilidade de IA: {overall['probability_ai']:.1f}%\n"
    output_text += f"   Confiança: {overall['confidence']}\n"
    output_text += f"   Conclusão: {overall['verdict']}\n\n"
    
  
    ela = results['error_level_analysis']
    output_text += f"📊 ANÁLISE DE NÍVEL DE ERRO (ELA)\n"
    output_text += f"   Erro Médio: {ela['mean_error']:.2f}\n"
    output_text += f"   Desvio Padrão: {ela['std_error']:.2f}\n"
    output_text += f"   Score de Uniformidade: {ela['uniformity_score']:.2%}\n"
    output_text += f"   Status: {'⚠️ Suspeito' if ela['suspicious'] else '✓ Normal'}\n\n"
    
   
    noise = results['noise_analysis']
    output_text += f"🔊 ANÁLISE DE RUÍDO\n"
    output_text += f"   Consistência de Ruído: {noise['noise_consistency']:.2%}\n"
    output_text += f"   Entropia: {noise['entropy']:.2f}\n"
    output_text += f"   Variância Laplacian: {noise['laplacian_variance']:.2f}\n"
    output_text += f"   Status: {'⚠️ Suspeito' if noise['suspicious'] else '✓ Normal'}\n\n"
    
   
    freq = results['frequency_analysis']
    output_text += f"📈 ANÁLISE DE FREQUÊNCIA\n"
    output_text += f"   Energia Baixa Frequência: {freq['low_frequency_energy']:.2e}\n"
    output_text += f"   Energia Alta Frequência: {freq['high_frequency_energy']:.2e}\n"
    output_text += f"   Razão de Frequência: {freq['frequency_ratio']:.2e}\n"
    output_text += f"   Status: {'⚠️ Suspeito' if freq['suspicious'] else '✓ Normal'}\n\n"
    
 
    comp = results['compression_analysis']
    output_text += f"🗜️ ANÁLISE DE COMPRESSÃO\n"
    output_text += f"   Formato: {comp['is_jpeg'] and 'JPEG' or 'Outro'}\n"
    output_text += f"   Artefatos de Compressão: {comp['compression_artifacts']:.2f}\n"
    output_text += f"   Status: {'⚠️ Suspeito' if comp['suspicious'] else '✓ Normal'}\n\n"
    
  
    meta = results['metadata_analysis']
    output_text += f"📋 ANÁLISE DE METADADOS\n"
    output_text += f"   Formato: {meta['format']}\n"
    output_text += f"   Modo de Cor: {meta['mode']}\n"
    output_text += f"   Tamanho: {meta['size']}\n"
    output_text += f"   Contém EXIF: {'Sim' if meta['has_exif'] else 'Não'}\n"
    output_text += f"   Status: {'⚠️ Suspeito' if meta['suspicious_metadata'] else '✓ Normal'}\n\n"
    
    output_text += "=" * 60 + "\n"
    output_text += "NOTA: Esta análise é baseada em técnicas de forense digital.\n"
    output_text += "Resultados devem ser interpretados como indicadores, não como\n"
    output_text += "prova definitiva de manipulação.\n"
    output_text += "=" * 60
    
    return output_text


def detect_ai_image(image_input):
   
    if image_input is None:
        return "Por favor, envie uma imagem para análise.", None
    
    detector = AIImageDetector()
    results, img_pil = detector.analyze_image(image_input)
    
    output_text = format_results(results, img_pil)
    
    return output_text, img_pil



with gr.Blocks(title="AI Image Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    
    IA IMAGE DETECTOR - BY KGB-LABS
    Este programa utiliza técnicas de análise forense digital para identificar:
    - **Análise de Nível de Erro (ELA)**: Detecta inconsistências de compressão
    - **Análise de Ruído**: Identifica padrões anormais de ruído
    - **Análise de Frequência**: Examina o espectro de frequência da imagem
    - **Análise de Compressão**: Detecta artefatos JPEG
    - **Análise de Metadados**: Verifica informações EXIF
    
    **Instruções**: Envie uma imagem para análise.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Envie a Imagem",
                type="pil",
                sources=["upload", "clipboard"]
            )
            analyze_btn = gr.Button(
                "🔍 Analisar Imagem",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Resultado da Análise",
                lines=30,
                interactive=False
            )
    
    with gr.Row():
        output_image = gr.Image(
            label="Imagem Analisada",
            type="pil"
        )
    
    # Conectar o botão à função
    analyze_btn.click(
        fn=detect_ai_image,
        inputs=image_input,
        outputs=[output_text, output_image]
    )
    
    gr.Markdown("""
    ---
    ### ⚠️ Aviso Importante
    - Esta ferramenta é para fins pesquisa
    - Os resultados não são 100% precisos e devem ser interpretados com cuidado pelo analista
    - Imagens reais podem ter características que as fazem parecer geradas por IA
    - Imagens geradas por IA evoluem constantemente e podem enganar detectores
    - Desenvolvida por KGB_LABS https://github.com/KGB-LABS arrudacibersec@proton.me

    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True
    )
