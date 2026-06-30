# Example face images

These four images are included so the application can be exercised immediately after installation.
Upload any pair through the two file uploaders in `app.py`.

## Suggested pairs

The following values were measured with the pinned InsightFace 1.0.1 Buffalo_L model and the
current calibration. They are examples, not guaranteed identity or resemblance claims.

| Image A | Image B | Purpose | Observed cosine | Observed score |
| --- | --- | --- | ---: | ---: |
| `victoria_justice_2018.png` | `nina_dobrev_2018.png` | Similar-looking different people | 0.1071 | 51 — High |
| `victoria_justice_2018.png` | `victoria_justice_2012.jpg` | Same person, different image | 0.5642 | 98 — Very high |
| `nina_dobrev_2018.png` | `nina_dobrev_2011.jpg` | Same person, different image | 0.4919 | 96 — Very high |

Scores can change with model, dependency, image-processing, or calibration updates. They are not
identity-verification probabilities.

## Attribution and licenses

All images came from Wikimedia Commons and are redistributed under their respective Creative
Commons licenses. The 800-pixel copies were obtained through Wikimedia's thumbnail service where
the original was larger. No endorsement by the subjects, photographers, or licensors is implied.

### `victoria_justice_2018.png`

- Subject: Victoria Justice
- Author: Ethan Sigmon; Commons cropped version by Luis fragar
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Victoria_Justice_%26_Madison_Justice_(cropped).png)
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

### `victoria_justice_2012.jpg`

- Subject: Victoria Justice
- Author: orangesporanges from Sydney
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Victoria_Justice_2012.jpg)
- License: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

### `nina_dobrev_2018.png`

- Subject: Nina Dobrev
- Author/attribution: MTV International
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Nina_Dobrev_during_an_interview_in_August_2018_02.png)
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

### `nina_dobrev_2011.jpg`

- Subject: Nina Dobrev
- Author: Gage Skidmore
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Nina_Dobrev_02.jpg)
- License: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

The photographs may also be subject to personality or publicity rights even though their
copyright licenses permit redistribution. Use them only in ways consistent with applicable law.
