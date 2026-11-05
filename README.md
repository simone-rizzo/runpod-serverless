# TACO litter segmentation — Runpod Serverless

Worker Serverless per un modello RF-DETR-Seg Medium addestrato su [TACO](https://github.com/pedropro/TACO)
(10 classi). I pesi non sono nell'immagine Docker: vengono letti dal network volume.

## Deploy

1. Pubblica questa cartella in un repository GitHub e autorizza Runpod in **Settings → Connections**.
2. **Serverless → New Endpoint → Import Git Repository**: scegli repo e branch, Dockerfile nella root,
   tipo di endpoint **Queue**.
3. In **Advanced → Network Volumes** seleziona il volume che contiene `taco_segmentation/`.
   L'endpoint potrà girare solo nel datacenter del volume, quindi scegli GPU disponibili lì.
4. **Deploy Endpoint**.

Un push non aggiorna l'endpoint: per ricostruire l'immagine bisogna creare una **GitHub release**.

## Percorso del modello

Lo stesso volume si vede in due punti diversi:

| Dove | Percorso |
| --- | --- |
| Pod | `/workspace/taco_segmentation/medium_12ep/checkpoint_best_total.pth` |
| Serverless | `/runpod-volume/taco_segmentation/medium_12ep/checkpoint_best_total.pth` |

## Variabili d'ambiente (tutte opzionali)

| Variabile | Default | Significato |
| --- | --- | --- |
| `MODEL_PATH` | percorso Serverless qui sopra | checkpoint da caricare |
| `DEFAULT_THRESHOLD` | `0.4` | confidenza minima se la richiesta non la specifica |
| `MAX_SIDE` | `640` | lato massimo dato al modello, lo stesso delle immagini di training |
| `MAX_DOWNLOAD_MB` | `25` | dimensione massima di `image_url` |

## API

Input, con esattamente uno tra `image` e `image_url`:

| Campo | Tipo | Default | Note |
| --- | --- | --- | --- |
| `image` | string | | base64 o data URI |
| `image_url` | string | | http o https |
| `threshold` | number | `0.4` | tra 0 e 1 |
| `return_masks` | boolean | `true` | `false` restituisce solo i box |

Il body di `/run` è limitato a 10 MB e quello di `/runsync` a 20 MB. Il base64 pesa circa un terzo in
più del file: per le foto più grandi usa `image_url`.

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_url": "https://img.tacodataset.org/img/33978196618.jpg"}}'
```

Output (dentro il campo `output` della risposta Runpod):

```json
{
  "image_width": 1537,
  "image_height": 2049,
  "threshold": 0.4,
  "inference_ms": 180,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Bottle",
      "confidence": 0.9627,
      "bbox_xyxy": [412.5, 630.1, 980.2, 1704.8],
      "mask_rle": {"size": [2049, 1537], "counts": "..."}
    }
  ]
}
```

I valori qui sopra sono illustrativi. Box e maschere sono in pixel dell'immagine inviata, dopo la
rotazione EXIF. Le detection sono ordinate per confidenza decrescente. Gli errori di input tornano come
`{"error": "..."}`.

`mask_rle` è in formato COCO RLE compresso:

```python
from pycocotools import mask as mask_utils

binary_mask = mask_utils.decode(detection["mask_rle"])  # array H x W di 0 e 1
```

## Test su un Pod

Con il volume montato su `/workspace` e le dipendenze installate:

```bash
MODEL_PATH=/workspace/taco_segmentation/medium_12ep/checkpoint_best_total.pth python handler.py
```

Il comando esegue `test_input.json` una volta e termina.
