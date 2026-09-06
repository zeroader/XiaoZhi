# Emotion emoji assets

Generated with the built-in imagegen tool. Source: `emotion_emoji.png`.
Run `scripts/prepare_vision_runtime_assets.py` to produce the eight 58×58 RGBA
runtime icons. The script crops to the face's alpha bounds, fits it within
56×56 pixels, and preserves the generated transparency.

Grid order (four columns, two rows):

| Label | Color |
| --- | --- |
| angry | Red-orange |
| contempt | Purple/magenta |
| disgust | Lime green |
| fear | Royal blue |
| happy | Golden yellow |
| neutral | Silver-gray |
| sad | Cyan/blue |
| surprise | Orange/amber |

Final edit prompt:

> Keep the exact eight emotions and four-column by two-row order. Give every
> emotion a distinct dominant color: angry red-orange, contempt purple/magenta,
> disgust lime/emerald green, fear royal blue; happy golden yellow, neutral
> silver-gray, sad cyan/deep blue, surprise orange/amber. Polished modern 3D
> emoji faces, expressive eyebrows/eyes/mouth, matching scale, clean silhouettes,
> transparent background, readable at 48–56 pixels. No words, labels, badges,
> people, bodies, watermark, overlap or cropping.
