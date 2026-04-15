# OpenVLA: An Open-Source Vision-Language-Action Model

著者: Moo Jin Kim∗,1, Karl Pertsch∗,1,2, Siddharth Karamcheti∗,1,3, Ted Xiao4, Ashwin Balakrishna3, Suraj Nair3, Rafael Rafailov1, Ethan Foster1, Grace Lam, Pannag Sanketi4, Quan Vuong5,†, Thomas Kollar3, Benjamin Burchfiel3, Russ Tedrake3,6, Dorsa Sadigh1, Sergey Levine2, Percy Liang1, Chelsea Finn1
リンク: https://openvla.github.io

### アブストラクト (Abstract)
インターネット規模の視覚-言語データと多様なロボットのデモンストレーションの組み合わせで事前学習された大規模な方策は、ロボットに新しいスキルを教える方法を変える可能性を秘めている。つまり、新しい行動をゼロから学習させるのではなく、そのような視覚-言語-行動 (VLA) モデルをファインチューニングすることで、視覚運動制御のための堅牢で汎化可能な方策を得ることができる。しかし、ロボティクスにおいてVLAを広く採用することは困難であった。なぜなら、1) 既存のVLAは大部分がクローズドであり、一般にアクセスできないこと、2) 先行研究では、採用の鍵となる新しいタスクに向けてVLAを効率的にファインチューニングする手法を探求していないこと、が挙げられる。これらの課題に対処するため、我々は97万件の実世界の多様なロボットのデモンストレーションのコレクションで学習された、パラメータ数70億のオープンソースVLAであるOpenVLAを導入する。OpenVLAは、DINOv2とSigLIPの事前学習済み特徴を融合する視覚エンコーダと組み合わせたLlama 2言語モデルをベースに構築されている。追加されたデータの多様性と新しいモデルコンポーネントの成果として、OpenVLAは汎用的なマニピュレーションにおいて強力な結果を示し、29のタスクと複数のロボットの身体(embodiment)にわたる絶対的なタスク成功率において、7分の1のパラメータ数でRT-2-X (550億パラメータ) などのクローズドモデルを16.5%上回った。さらに、複数の物体が関与するマルチタスク環境において特に強力な汎化結果と強力な言語グラウンディング能力を示し、新しい設定に対してOpenVLAを効果的にファインチューニングできることを示し、Diffusion Policyのような表現力の高いゼロからの模倣学習手法を20.4%上回った。我々は計算効率についても探求しており、別の貢献として、OpenVLAが最新の低ランク適応 (LoRA) 手法によってコンシューマー向けGPU上でファインチューニングでき、下流タスクの成功率を下げることなく量子化によって効率的に提供できることを示す。最後に、モデルのチェックポイント、ファインチューニング用のノートブック、およびOpen X-Embodimentデータセット上でVLAを大規模に学習するためのサポートが組み込まれたPyTorchコードベースを公開する。

---

### 1 はじめに (Introduction)
ロボットマニピュレーションのための学習済み方策の主な弱点は、学習データを超えて汎化する能力がないことである。個別のスキルや言語指示のために学習された既存の方策は、物体の位置や照明などの新しい初期条件に対して行動を外挿する能力を持っているが [^2], [^3]、シーンの気晴らし(distractor)や新しい物体に対する堅牢性を欠いており [^4], [^5]、未経験のタスク指示を実行するのに苦労する [^6], [^7]。しかし、ロボティクスの分野を超えて、CLIP [^8]、SigLIP [^9]、Llama 2 [^10] などの視覚と言語のための既存の基盤モデルは、インターネット規模の事前学習データセットによって捉えられた事前知識に由来して、これらのタイプの汎化やそれ以上のことが可能である。ロボティクスのためにこの規模の事前学習を再現することは依然として未解決の課題であり（最大規模のロボットマニピュレーションデータセット [^1], [^11] であっても、10万から100万の例しか持たない）、この不均衡はある機会を示唆している。すなわち、学習データを超えて物体、シーン、タスクに汎化できるロボットの方策を学習するためのコアな構成要素として、視覚と言語のための既存の基盤モデルを利用することである。

この目標に向けて、既存の研究は、ロボットの表現学習のために事前学習済みの言語および視覚-言語モデルを統合すること [^12], [^13], [^14] や、タスク計画および実行のためのモジュールシステムのコンポーネントとして統合すること [^15], [^16] を探求してきた。より最近では、制御のために視覚-言語-行動モデル (VLAs) [^1], [^7], [^17], [^18] を直接学習するために使用されている。VLAは、ロボット制御の行動を生成するためにPaLI [^19], [^20] のような視覚条件付き言語モデル (VLMs) を直接ファインチューニングすることで、ロボティクスに事前学習済みの視覚および言語の基盤モデルを使用する直接的なインスタンス化を提供する。インターネット規模のデータで学習された強力な基盤モデルを土台とすることで、RT-2 [^7] のようなVLAは、印象的な堅牢性の結果と、新しい物体やタスクに汎化する能力を実証し、汎用的なロボット方策の新たな基準を確立している。しかし、既存のVLAの普及を妨げている2つの主な理由がある。1) 現在のモデル [^1], [^7], [^17], [^18] はクローズドであり、モデルアーキテクチャ、学習手順、データ混合についての可視性が限られていること。2) 既存の研究では、特にコモディティハードウェア（例：コンシューマー向けのGPU）上で、新しいロボット、環境、タスクにVLAを展開し適応させるためのベストプラクティスが提供されていないことである。我々は、将来の研究開発のための豊かな基盤を構築するために、ロボティクスには、オープンソース言語モデルの既存のエコシステム [^21], [^22], [^23], [^24] に似た、効果的なファインチューニングと適応をサポートするオープンソースで汎用的なVLAが必要であると主張する。

この目的のために、汎用的なロボットマニピュレーション方策の新しい最高水準を確立する、パラメータ数70億のオープンソースVLAであるOpenVLAを導入する[^1注]。OpenVLAは、複数の粒度で視覚的特徴を捉える事前学習済みの視覚条件付き言語モデルのバックボーンで構成されており、幅広いロボットの身体、タスク、シーンを網羅するデータセットであるOpen-X Embodiment [^1] データセットから得られた97万件のロボットマニピュレーション軌跡の、大規模で多様なデータセットでファインチューニングされている。データの多様性の向上と新しいモデルコンポーネントの成果として、OpenVLAは、WidowXおよびGoogle Robotの身体での29の評価タスクにわたり、絶対的な成功率で、以前の最高水準のVLAであった550億パラメータのRT-2-Xモデル [^1], [^7] を16.5%上回った。我々はさらに、先行研究では探求されていなかった新たな貢献として、物体のピック・アンド・プレースからテーブルの清掃までの行動にまたがる7つの多様なマニピュレーションタスクにわたって、VLAのための効率的なファインチューニング戦略を調査する。ファインチューニングされたOpenVLAの方策は、Octo [^5] のようなファインチューニングされた事前学習済み方策を明らかに上回ることがわかった。Diffusion Policy [^3] によるゼロからの模倣学習と比較して、ファインチューニングされたOpenVLAは、複数の物体があるマルチタスク環境において行動に言語をグラウンディングさせるタスクにおいて大幅な改善を示す。これらの結果に続いて、我々は初めて、パフォーマンスを損なうことなく、大規模なサーバーノードの代わりにコンシューマー向けGPU上でOpenVLAモデルを適応させることを容易にするための、低ランク適応 (LoRA) [^26] とモデル量子化 [^27] を活用した計算効率の高いファインチューニング手法の有効性を実証する。最後の貢献として、我々はすべてのモデル、展開およびファインチューニング用のノートブック、そしてOpen X-Embodimentデータセット上でVLAを大規模に学習するためのOpenVLAコードベースをオープンソース化し、これらのリソースがロボティクスのためにVLAを探求し適応させる将来の研究を可能にすることを期待している。

[^1注]: OpenVLAは、SigLIP [^9] とDinoV2 [^25] の視覚エンコーダ、そしてLlama 2 [^10] の言語モデルのバックボーンという複数の事前学習済みモデルコンポーネントを使用している。これら3つのモデルについては重みはオープンだが、学習データやコードはオープンではない。我々は、これらのコンポーネントの上にOpenVLAを再現するための学習データ、コード、モデルの重みを公開する。

(1) 強力なオープンVLMのバックボーンをより豊富なロボット事前学習データセットと組み合わせることで、OpenVLAは1桁小さい（550億パラメータに対して70億）にもかかわらず、我々の実験においてRT-2-Xを上回る。(2) 我々は新しいターゲット設定へのOpenVLAモデルのファインチューニングを徹底的に調査するが、RT-2-Xはファインチューニングの設定を調査していない。(3) 我々は初めて、VLAに対する最新のパラメータ効率の良いファインチューニングと量子化アプローチの有効性を実証する。(4) OpenVLAはオープンソースである初の汎用VLAであり、したがってVLAの学習、データ混合、目的関数、推論に関する将来の研究をサポートする。

---

### 2 関連研究 (Related Work)
**視覚条件付き言語モデル (Visually-Conditioned Language Models)** インターネット規模のデータで学習され、入力画像と言語のプロンプトから自然言語を生成する視覚条件付き言語モデル (VLM) は、視覚的質問応答 [^28], [^29], [^30], [^31] から物体ローカリゼーション [^32], [^33] まで無数のアプリケーションに採用されてきた。最近のVLMを推進する重要な進歩の1つは、事前学習済みの視覚エンコーダ [^8], [^9], [^25] からの特徴を事前学習済みの言語モデル [^10], [^23], [^34], [^35], [^36] と橋渡しするモデルアーキテクチャであり、コンピュータビジョンと自然言語モデリングの両方の進歩を直接構築して強力なマルチモーダルモデルを作成している。初期の研究では視覚と言語の特徴の間でクロスアテンションを行うための様々なアーキテクチャが探求されたが [^37], [^38], [^39], [^40], [^41]、新しいオープンソースのVLM [^20], [^42], [^43], [^44] はよりシンプルな「パッチをトークンとして扱う (patch-as-token)」アプローチに収束しており、そこでは事前学習済みの視覚Transformerからのパッチ特徴がトークンとして扱われ、言語モデルの入力空間に射影される。このシンプルさにより、言語モデルを大規模に学習するための既存のツールをVLMの学習に容易に転用できる。我々は本研究において、VLAの学習をスケールアップするためにこれらのツールを採用し、特にKaramchetiら [^44] のVLMを事前学習済みのバックボーンとして使用する。なぜなら、それらはマルチ解像度の視覚特徴から学習され、視覚的汎化を助けるためにDINOv2 [^25] からの低レベルの空間情報とSigLIP [^9] からの高レベルの意味論を融合しているからである。

**汎用ロボット方策 (Generalist Robot Policies)** ロボティクスにおける最近のトレンドは、多くの異なるロボットの身体 [^1], [^5], [^53], [^57], [^58], [^59], [^60], [^61], [^62], [^63], [^64], [^65], [^66] を網羅する、大規模で多様なロボットデータセット [^1], [^2], [^6], [^11], [^45], [^49], [^50], [^51], [^52], [^53], [^54], [^55], [^56] の上で、マルチタスクの「汎用」ロボット方策 [^2], [^6], [^45], [^46], [^47], [^48], [^49] を学習することに向かっている。注目すべきことに、Octo [^5] は、そのまま複数のロボットを制御でき、新しいロボットのセットアップへの柔軟なファインチューニングを可能にする汎用方策を学習する。これらのアプローチとOpenVLAの大きな違いはモデルアーキテクチャである。Octoのような先行研究は通常、言語埋め込みや視覚エンコーダのような事前学習済みコンポーネントを、ゼロから初期化された追加のモデルコンポーネント [^2], [^5], [^6] と構成し、方策の学習過程でそれらを「縫い合わせる (stitch)」ことを学習する。これらの研究とは異なり、OpenVLAはよりエンドツーエンドなアプローチを採用し、ロボットの行動を言語モデルの語彙のトークンとして扱うことで、ロボットの行動を予測するためにVLMを直接ファインチューニングする。我々の実験的評価は、このシンプルかつスケーラブルなパイプラインが、従来の汎用方策に比べて性能と汎化能力を大幅に高めることを示している。

**視覚-言語-行動モデル (Vision-Language-Action Models)** 多くの研究がロボティクスにおけるVLMの使用を探求してきた。例えば、視覚的な状態表現 [^12], [^13]、物体検出 [^67]、ハイレベルなプラニング [^16]、およびフィードバック信号の提供 [^68], [^69], [^70], [^71] などである。その他の研究は、VLMをエンドツーエンドの視覚運動マニピュレーション方策 [^14], [^15] に直接統合しているが、方策のアーキテクチャに大きな構造を組み込んだり、キャリブレーションされたカメラを必要としたりするため、その適用性は制限される。最近の多くの研究は、我々と同様の手法を探求し、ロボットの行動を予測するために大規模な事前学習済みVLMを直接ファインチューニングしている [^1], [^7], [^17], [^18], [^72], [^73], [^74]。そのようなモデルはしばしば視覚-言語-行動モデル (VLAs) と呼ばれる。なぜなら、それらはロボットの制御行動をVLMのバックボーンに直接融合させるからである。これには3つの大きな利点がある。(1) 大規模なインターネット規模の視覚-言語データセット上で、事前学習された視覚と構成要素の配置 (alignment) を実行する。(2) ロボット制御専用に作られていない汎用アーキテクチャを使用することで、最新のVLM学習 [^75], [^76], [^77] の基礎となるスケーラブルなインフラストラクチャを活用でき、最小限のコード修正で数十億パラメータの方策の学習にスケールできる。(3) VLMの急速な改良からロボティクスが恩恵を受けるための直接的な経路を提供する。VLAに関する既存の研究は、単一のロボットまたはシミュレーション環境での学習と評価に焦点を当てているため汎用性を欠いているか [^72], [^73], [^74], [^78]、またはクローズドであり新しいロボットのセットアップへの効率的なファインチューニングをサポートしていない [^1], [^7], [^17], [^18]。最も密接に関連するものとして、RT-2-X [^1] はOpen X-Embodimentデータセット上で550億パラメータのVLA方策を学習し、最高水準の汎用マニピュレーション方策の性能を実証している。しかし、我々の研究は以下の重要な点でRT-2-Xとは異なる。(1) 強力なオープンVLMバックボーンとより豊富なロボット事前学習データセットを組み合わせることで、OpenVLAは実験においてRT-2-Xを上回りつつ、1桁小さいサイズである。(2) 我々はOpenVLAモデルの新しいターゲット設定へのファインチューニングを徹底的に調査するのに対し、RT-2-Xはファインチューニングの設定を調査していない。(3) 我々はVLAに対する最新のパラメータ効率の良いファインチューニングと量子化アプローチの有効性を初めて実証する。(4) OpenVLAはオープンソースである初の汎用VLAであり、VLAの学習、データ混合、目的関数、推論に関する将来の研究をサポートする。

---

### 3 OpenVLA モデル (The OpenVLA Model)
我々は、Open X-Embodimentデータセット [^1] からの97万件のロボットのデモンストレーションで学習された、パラメータ数70億の視覚-言語-行動モデル (VLA) であるOpenVLAモデルを導入する。VLAモデルを開発するためのベストプラクティスについては、例えば学習に使用する最適なモデルのバックボーン、データセット、ハイパーパラメータなど、未解明の疑問が多く存在している。以下では、OpenVLAを開発するための我々のアプローチを詳述し、得られた主要な知見を要約する。具体的には、まずOpenVLAのバックボーンを形成する最新のVLMの概要を簡単に説明し (セクション3.1)、次に我々の基本的な学習レシピとデータセットについて説明し (セクション3.2および3.3)、重要な設計上の決定について議論し (セクション3.4)、学習と推論に使用したインフラストラクチャの詳細を提供する (セクション3.5)。

#### 3.1 準備：視覚-言語モデル (Preliminaries: Vision-Language Models)
最近のほとんどのVLM [^20], [^42], [^43], [^44] のアーキテクチャは、3つの主要な部分から構成されている (図2を参照)：(1) 画像入力をいくつかの「画像パッチ埋め込み」にマッピングする視覚エンコーダ、(2) 視覚エンコーダの出力埋め込みを受け取り、それらを言語モデルの入力空間にマッピングするプロジェクタ、および(3) 大規模言語モデル (LLM) のバックボーン。VLMの学習中、モデルは、インターネットの様々なソースから収集された、ペアになった、またはインターリーブされた視覚と言語のデータに対して、次のテキストトークンを予測する目的関数でエンドツーエンドに学習される。本研究において、我々はPrismatic-7B VLM [^44] をベースに構築する。Prismaticは、6億パラメータの視覚エンコーダ、小さな2層のMLPプロジェクタ、および70億パラメータのLlama 2言語モデルバックボーン [^10] を備えた、上記で説明したのと同じ標準的なアーキテクチャに従っている。特に、Prismaticは事前学習済みのSigLIP [^79] およびDinoV2 [^25] モデルで構成される2部構成の視覚エンコーダを使用する。入力画像パッチは両方のエンコーダを別々に通過し、得られた特徴ベクトルはチャネルごとに連結される。CLIP [^80] またはSigLIPのみのエンコーダなど、より一般的に使用される視覚エンコーダとは対照的に、DinoV2の特徴を追加することは、ロボット制御に特に役立つ可能性のある空間推論の改善に役立つことが示されている [^44]。SigLIP、DinoV2、およびLlama 2は、それらの学習データの詳細を公開していないが、おそらくそれぞれ数兆トークンに及ぶインターネット由来の画像-テキスト、画像のみ、およびテキストのみのデータで構成されていると思われる。Prismatic VLMは、LLaVA 1.5データ混合 [^43] を使用してこれらのコンポーネント上でファインチューニングされている。これには、オープンソースデータセット [^29], [^42], [^81], [^82], [^83] からの合計約100万件の画像-テキストおよびテキストのみのデータサンプルが含まれている。

#### 3.2 OpenVLAの学習手順 (OpenVLA Training Procedure)
OpenVLAを学習させるために、我々はロボットの行動予測のために事前学習済みのPrismatic-7B VLMバックボーンをファインチューニングする (図2を参照)。我々は行動予測問題を「視覚-言語」タスクとして定式化し、入力観測画像と自然言語のタスク指示が、予測されたロボットの行動の文字列にマッピングされる [^7]。VLMの言語モデルバックボーンがロボットの行動を予測できるようにするため、連続的なロボットの行動を言語モデルのトークナイザが使用する離散的なトークンにマッピングすることにより、LLMの出力空間に行動を表現する。Brohanら [^7] に従い、我々はロボットの行動の各次元を別々に256個のビンのいずれかに離散化する。各行動の次元について、学習データ内の行動の1パーセンタイルから99パーセンタイルの間の区間を均等に分割するようにビン幅を設定する。Brohanら [^7] が使用した最小-最大境界の代わりに分位点を使用することで、離散化区間を大幅に拡大し、行動の離散化の実効的な粒度を低下させる可能性のあるデータ内の外れ値の行動を無視することができる。

この離散化を使用すると、 $N$ 次元のロボットの行動について、 $N$ 個の離散的な整数 $\in [0 \dots 255]$ が得られる。あいにく、OpenVLAの言語バックボーンが使用するトークナイザであるLlamaトークナイザ [^10] は、ファインチューニング中に新しく導入されるトークン用に100個の「特殊トークン」しか予約しておらず、これは我々の行動の離散化の256トークンには少なすぎる。代わりに、我々は再びシンプルさを選び、Brohanら [^7] のアプローチに従い、Llamaトークナイザの語彙の中で最も使用頻度の低い256個のトークン（最後の256個のトークンに相当）を単に行動トークンで上書きする。行動がトークンのシーケンスに処理された後、OpenVLAは、予測された行動トークンのみについてクロスエントロピー損失を評価する標準的な次のトークン予測目的関数で学習される。この学習手順を実装するための重要な設計上の決定についてはセクション3.4で議論する。次に、OpenVLAの学習に使用するロボットのデータセットについて説明する。

#### 3.3 学習データ (Training Data)
OpenVLAの学習データセットを構築する目的は、ロボットの身体、シーン、およびタスクの大きな多様性を捉えることである。これにより、最終的なモデルは、そのままの状態で様々なロボットを制御することが可能になり、新しいロボットのセットアップへの効率的なファインチューニングを許容する。我々はOpen X-Embodimentデータセット [^1] (OpenX) をベースとして活用し、学習データセットをキュレーションする。執筆時点での完全なOpenXデータセットは、70を超える個別のロボットデータセットで構成されており、大規模なコミュニティの取り組みにおいて、一貫性があり使いやすいデータ形式にプールされた200万を超えるロボットの軌跡が含まれている。このデータでの学習を実用的なものにするため、元のデータセットにデータキュレーションの複数のステップを適用する。

このキュレーションの目的は、(1) すべての学習データセットにわたって一貫した入力および出力空間を確保すること、および(2) 最終的な学習の混合物において身体、タスク、およびシーンのバランスの取れた混合を確保することである[^2注]。(1)に対処するため、我々は [^1], [^5] に従い、学習データセットを少なくとも1つの三人称カメラを持ち、単腕のエンドエフェクタ制御を使用するマニピュレーションデータセットのみを含むように制限する。(2)については、最初のフィルタリングを通過したすべてのデータセットに対してOcto [^5] のデータ混合の重みを活用する。Octoは、多様性の低いデータセットの重みをヒューリスティックに下げるか削除し、タスクやシーンの多様性が大きいデータセットの重みを上げる。詳細についてはOcto Model Teamら [^5] を参照されたい。

[^2注]: Octo [^5] は、異種感覚入力を持つデータセットにわたる学習を実証した。これは非常に有望であるが、異種のセンサーモダリティと行動空間にわたるVLAの学習の調査は将来の研究に残す。

我々はまた、控えめな混合重み10%ではあるが、Octoのリリース以降にOpenXデータセットに追加された、DROIDデータセット [^11] を含むいくつかの追加データセットを学習の混合物に組み込む実験を行った。実際には、DROIDの行動トークンの精度は学習を通じて低いままであることが判明し、その多様性に適合するには将来的に大きな混合重みやモデルが必要になる可能性が示唆された。最終的なモデルの品質を危険にさらさないために、我々は学習の最後の3分の1についてデータ混合からDROIDを削除した。使用したデータセットと混合の重みの完全な概要については付録Aで提供する。

#### 3.4 OpenVLAの設計上の決定 (OpenVLA Design Decisions)
OpenVLAモデルを開発する際、反復のスピードを上げ、計算コストを削減するために、最終的なモデルの学習実行を開始する前に、完全なOpenXの混合で学習するのではなく、小規模な実験で様々な設計上の決定を調査した。具体的には、我々は初期の実験としてBridgeData V2 [^6] 上でOpenVLAモデルを学習し評価した。以下に、これらの調査から得られた主要な知見を要約する。

**VLMのバックボーン (VLM Backbone).** 当初、我々は複数のVLMバックボーンを実験した。Prismatic [^44] とは別に、IDEFICS-1 [^84] とLLaVA [^85] をロボット行動予測のためにファインチューニングするテストを行った。LLaVAとIDEFICS-1はシーンに物体が1つしかないタスクでは同等のパフォーマンスを示したが、LLaVAは、シーンに複数の物体が存在し、方策が正しい物体（すなわち言語指示で指定された物体）を操作することを要求されるタスクにおいて、より強力な言語グラウンディングを実証した。具体的には、LLaVAはBridgeData V2のシンク環境での5つの言語グラウンディングタスクの平均で、IDEFICS-1を絶対成功率で35%上回った。

ファインチューニングされたPrismatic VLMの方策はさらに改善を達成し、シンプルな単一物体のタスクと複数の物体の言語グラウンディングタスクの両方にわたって、LLaVAの方策を絶対成功率で約10%上回った。我々はこの性能の差を、融合されたSigLIP-DinoV2のバックボーンによってもたらされる空間推論能力の向上に起因すると考える（セクション3.1を参照）。性能向上に加えて、Prismaticはモジュール化された使いやすいコードベースも提供しているため、我々は最終的にそれをOpenVLAモデルのバックボーンとして選択した。

**画像解像度 (Image Resolution).** 高解像度の画像はより多くの画像パッチトークンをもたらし、その結果コンテキスト長が長くなって学習の計算量が二次的に増加するため、入力画像の解像度はVLA学習の計算要件に大きな影響を与える。我々は $224 \times 224$ pxと $384 \times 384$ pxの入力を持つVLAを比較したが、後者は学習に3倍の時間がかかる一方で、我々の評価ではパフォーマンスの違いは見られなかった。

したがって、最終的なOpenVLAモデルでは $224 \times 224$ pxの解像度を選択する。多くのVLMのベンチマークでは解像度の向上がパフォーマンスを向上させるが [^44], [^86], [^87]、我々はVLAについては（まだ）この傾向を確認していないことに注意されたい。

**視覚エンコーダのファインチューニング (Fine-Tuning Vision Encoder).** VLMに関する先行研究では、VLMの学習中に視覚エンコーダを凍結することが一般的に高いパフォーマンスにつながることがわかっている [^44]。直感的には、凍結された視覚エンコーダは、インターネット規模の事前学習から学習された堅牢な特徴をより良く保持するかもしれない。しかし、我々はVLAの学習中に視覚エンコーダをファインチューニングすることが、良好なVLAのパフォーマンスのために重要であることを発見した。我々は、事前学習された視覚バックボーンは、正確なロボット制御を可能にするために、シーンの重要な部分に関するきめ細かい空間的詳細を十分に捉えきれていないのではないかという仮説を立てている。

**学習エポック数 (Training Epochs).** 通常のLLMやVLMの学習実行は、学習データセットを通過するエポックが最大でも1〜2エポックで完了する。

対照的に、VLAの学習では学習データセットを何度も繰り返し処理することが重要であり、学習における行動トークンの精度が95%を超えるまで実機ロボットのパフォーマンスは継続して向上することがわかった。我々の最終的な学習実行は、学習データセットを27エポック通過して完了する。

**学習率 (Learning Rate).** 我々はVLAの学習のために複数桁にわたって学習率をスイープし、 $2 \times 10^{-5}$ （VLMの事前学習時に使用されたのと同じ学習率 [^44]）の固定学習率を使用した場合に最高の結果を達成した。我々は学習率のウォームアップがメリットをもたらすことは発見できなかった。

#### 3.5 学習と推論のためのインフラストラクチャ (Infrastructure for Training and Inference)
最終的なOpenVLAモデルは、バッチサイズ2048を使用して、64基のA100 GPUのクラスタで14日間、合計21,500 A100時間で学習される。

推論中、OpenVLAはbfloat16精度で（すなわち量子化なしで）ロードされた場合に15GBのGPUメモリを必要とし、1台のNVIDIA RTX 4090 GPU上で（コンパイル、投機的デコード、またはその他の推論スピードアップのトリックなしで）約6Hzで実行される。セクション5.4で示すように、実世界のロボティクスタスクでのパフォーマンスを損なうことなく、量子化を介して推論中のOpenVLAのメモリフットプリントをさらに削減することができる。我々は様々なコンシューマー向けおよびサーバー向けGPUでの推論速度を図6に報告する。利便性のため、ロボットへの行動予測のリアルタイムなリモートストリーミングを可能にするリモートVLA推論サーバーを実装し、ロボットを制御するための強力なローカル計算デバイスへのアクセスを要求する要件を取り除いている。我々はこのリモート推論ソリューションをオープンソースのコードリリースの一部として公開する（セクション4）。

---

### 4 OpenVLAコードベース (The OpenVLA Codebase)
我々のモデルと共に、我々はVLAモデルを学習するためのモジュール式PyTorchコードベースであるOpenVLAコードベースをリリースする（https://openvla.github.io を参照）。これは、個々のGPU上でのVLAのファインチューニングから、マルチノードのGPUクラスタでの数十億パラメータのVLAの学習までスケールし、大規模なTransformerモデルの学習のための自動混合精度 (AMP、PyTorch [^75])、FlashAttention [^76]、および完全シャードデータ並列処理 (FSDP、Zhaoら [^77]) などの最新の技術をサポートしている。そのままの状態で、OpenVLAコードベースはOpen Xデータセットでの学習を完全にサポートし、HuggingFace [^21] のAutoModelクラスと統合されており、LoRAファインチューニング [^26] と量子化されたモデルの推論 [^27], [^88] をサポートしている。

---

図1: 我々は、Open X-Embodimentデータセット [^1] からの97万件のロボットエピソードで学習された、パラメータ数70億のオープンソース視覚-言語-行動モデル (VLA) であるOpenVLAを提示する。OpenVLAは、汎用ロボットマニピュレーション方策の新たな最高水準を確立する。そのままの状態で複数のロボットの制御をサポートし、パラメータ効率の良いファインチューニングを介して新しいロボットのドメインに素早く適応することができる。OpenVLAのチェックポイントとPyTorchの学習パイプラインは完全にオープンソースであり、モデルはHuggingFaceからダウンロードしてファインチューニングすることができる。

図2: OpenVLAのモデルアーキテクチャ。画像観測と言語指示が与えられると、モデルは7次元のロボット制御行動を予測する。このアーキテクチャは3つの主要なコンポーネントで構成されている。(1) Dino V2 [^25] とSigLIP [^79] の特徴を連結する視覚エンコーダ、(2) 視覚特徴を言語の埋め込み空間にマッピングするプロジェクタ、および(3) LLMバックボーンであるLlama 2の70億パラメータの大規模言語モデル [^10]。

図3: BridgeData V2 WidowXロボットの評価タスクと結果。我々はOpenVLAと従来の最高水準の汎用ロボット方策を、言語条件付け能力を特別に評価するタスクだけでなく、汎化のいくつかの軸をカバーするタスクの包括的なスイートで評価する。OpenVLAは全体で最高のパフォーマンスを達成し、意味論的汎化(semantic generalization)を除くすべてのカテゴリーで、クローズドソースのモデルであるRT-2-Xをも上回る。平均成功率±標準誤差(StdErr)は、各アプローチにつき合計170のロールアウトにわたって計算されている。詳細な結果については表4を参照。

図4: Googleロボットの評価結果。我々は、RT-1およびRT-2の評価 [^2], [^7] で使用されたモバイルマニピュレータ上で、分布内および分布外 (OOD) のタスクについて汎用ロボット方策を評価する。OpenVLAとRT-2-Xは同等のパフォーマンスを達成し、全体としてRT-1-XとOctoを大幅に上回ることがわかった。平均成功率±標準誤差(StdErr)は、各アプローチにつき合計60のロールアウトにわたって計算されている。詳細な結果については表6を参照。

図5: 新しいロボットセットアップへの適応。我々は、7つのFranka Emika Pandaのタスク（各タスクにつき10〜150のデモンストレーション）でゼロから学習された最高水準のDiffusion Policyと、同じデータでファインチューニングされた汎用ロボット方策であるOctoおよびOpenVLAを評価する。Diffusion Policyは狭い単一指示のタスクで強力なパフォーマンスを示す一方で、OctoとOpenVLAは複数の指示と気晴らしの物体(distractor objects)を含む多様なファインチューニングのタスクでより良いパフォーマンスを示す。全体として、OpenVLAは両方のセットアップで最高の集計パフォーマンスを達成しており、下流タスクの方策を学習するための効果的なデフォルトになり得ることを示唆している。平均成功率±標準誤差(StdErr)は、各アプローチにつき129のロールアウト（Franka-Tabletopタスクでは99、Franka-DROIDタスクでは30）にわたって計算されている。詳細な結果については表7を参照。

図6: 様々なGPUでのOpenVLAの推論速度。bfloat16およびint4の量子化は、特にAda LovelaceアーキテクチャのGPU (RTX 4090, H100) 上で高いスループットを達成する。TensorRT-LLM [^89] のような最新のLLM推論フレームワークを用いることで、さらなるスピードアップが可能である。♠: モデルは収めるために2つのGPUに分割されている。

表1: パラメータ効率の良いファインチューニングの評価。LoRAによるファインチューニングは最高のパフォーマンスと計算のトレードオフを達成し、モデルのパラメータのわずか1.4%を学習するだけで、完全なファインチューニングのパフォーマンスに匹敵する。平均成功率±標準誤差(StdErr)は、選ばれたFranka-Tabletopタスク（詳細は表8を参照）での各アプローチにつき33のロールアウトにわたって計算されている。∗: FSDP [^77] を用いて2つのGPUに分割されている。

| Strategy | Success Rate | Train Params ($\times 10^6$) | VRAM (batch 16) |
| --- | --- | --- | --- |
| Full FT | 69.7 ± 7.2 % | 7,188.1 | 163.3 GB* |
| Last layer only | 30.3 ± 6.1 % | 465.1 | 51.4 GB |
| Frozen vision | 47.0 ± 6.9 % | 6,760.4 | 156.2 GB* |
| Sandwich | 62.1 ± 7.9 % | 914.2 | 64.0 GB |
| LoRA, rank=32 | 68.2 ± 7.5% | 97.6 | 59.7 GB |
| rank=64 | 68.2 ± 7.8% | 195.2 | 60.5 GB |

表2: 量子化された推論でのパフォーマンス。4-bit量子化は、GPUのメモリフットプリントを半分未満に削減しながら、bfloat16推論（我々のデフォルトアプローチ）のパフォーマンスに匹敵する。平均成功率±標準誤差(StdErr)は、8つの代表的なBridgeData V2のタスク [^6] および各アプローチにつき80のロールアウト（詳細は表5を参照）にわたって計算されている。

| Precision | Bridge Success | VRAM |
| --- | --- | --- |
| bfloat16 | 71.3 ± 4.8% | 16.8 GB |
| int8 | 58.1 ± 5.1% | 10.2 GB |
| int4 | 71.9 ± 4.7% | 7.0 GB |

### 5 実験 (Experiments)
我々の実験的評価の目標は、OpenVLAがそのままの状態で強力なマルチロボット制御方策として機能する能力や、新しいロボットのタスクへのファインチューニングのための良い初期化となる能力をテストすることである。具体的には、我々は以下の疑問に答えることを目指す：

1. 複数のロボットや様々なタイプの汎化能力を評価する際、OpenVLAはこれまでの汎用ロボット方策と比べてどうか？
2. OpenVLAは新しいロボットのセットアップやタスクに対して効果的にファインチューニングできるか？また、最先端のデータ効率の良い模倣学習アプローチと比べてどうか？
3. OpenVLAモデルの学習と推論の計算要件を減らし、よりアクセスしやすくするために、パラメータ効率の良いファインチューニングや量子化を使用できるか？パフォーマンスと計算量のトレードオフはどのようなものか？

#### 5.1 複数のロボットプラットフォームでの直接評価 (Direct Evaluations on Multiple Robot Platforms)
**ロボットのセットアップとタスク (Robot Setups and Tasks).** 我々は、BridgeData V2の評価 [^6] からのWidowXロボット（図1の左を参照）と、RT-1およびRT-2の評価 [^2], [^7] からのモバイルマニピュレーションロボット（「Googleロボット」、図1の中央を参照）という2つのロボットの身体について、OpenVLAの「そのままの状態 (out-of-the-box)」でのパフォーマンスを評価する。両方のプラットフォームは、汎用ロボット方策を評価するための先行研究 [^1], [^2], [^5], [^7] において広範に使用されてきた。我々は各環境において、視覚的汎化（未経験の背景、気晴らしの物体、物体の色や外観）、動作的汎化（未経験の物体の位置や向き）、物理的汎化（未経験の物体のサイズや形状）、および意味論的汎化（未経験のターゲット物体、指示、インターネットからの概念）といった、様々な汎化の軸をカバーする包括的な評価タスクのセットを定義する。また、複数の物体があるシーンにおける言語条件付けの能力を評価し、ユーザーのプロンプトで指定された通りに方策が正しいターゲット物体を操作できるかをテストする。BridgeData V2およびGoogleロボットの評価におけるタスク画像の例については、それぞれ図3および図4の一番下の行を参照されたい。全体として、我々は各手法をBridgeData V2の実験では170回のロールアウト（17タスク×10回の試行）、Googleロボットの実験では60回のロールアウト（12タスク×5回の試行）で評価した。すべてのタスクの詳細な内訳と、それらが学習データとどのように異なるかについては付録Bに記載されている。このセクションおよび以下のセクションにおけるすべての評価は、公平な比較を確実にするために、同じタスク、同じロボットおよび物体の初期状態のセットを使用したA/B評価として実施される。

図3：BridgeData V2 WidowXロボットの評価タスクと結果。我々はOpenVLAと従来の最高水準の汎用ロボット方策を、汎化のいくつかの軸をカバーする包括的なタスクのスイートだけでなく、言語条件付け能力を特別に評価するタスクで評価する。OpenVLAは全体として最高のパフォーマンスを達成し、意味論的汎化を除くすべてのカテゴリーでクローズドソースのモデルであるRT-2-Xをも上回る。平均成功率±標準誤差(StdErr)は、各アプローチにつき合計170のロールアウトにわたって計算されている。詳細な結果については表4を参照。

図4：Googleロボットの評価結果。我々はRT-1およびRT-2の評価 [^2], [^7] で使用されたモバイルマニピュレータ上で、分布内および分布外 (OOD) のタスクについて汎用ロボット方策を評価する。我々はOpenVLAとRT-2-Xが同等のパフォーマンスを達成し、全体としてRT-1-XとOctoを大幅に上回ることを発見した。平均成功率±標準誤差(StdErr)は、各アプローチにつき合計60のロールアウトにわたって計算されている。詳細な結果については表6を参照。

**比較 (Comparisons).** 我々はOpenVLAのパフォーマンスを、これまでの3つの汎用マニピュレーション方策であるRT-1-X [^1]、RT-2-X [^1]、およびOcto [^5] と比較する。RT-1-X（パラメータ数3500万）とOcto（パラメータ数9300万）は、OpenXデータセットのサブセット上でゼロから学習されたTransformerの方策である。Octoはオープンソースのマニピュレーション方策の中で最高水準のモデルである。RT-2-X（パラメータ数550億）は、インターネットで事前学習された視覚および言語のバックボーンを活用した最高水準のクローズドソースのVLAである。結果はBridgeData V2の評価については図3に、Googleロボットの評価については図4にまとめられている（タスクごとの内訳は付録の表4および表6を参照）。我々は、RT-1-XとOctoの両方がテストされたタスクで苦戦しており、特に気晴らしの物体が存在する場合には正しい物体を操作することに頻繁に失敗し、場合によってはロボットが腕をあてもなく振り回す原因となることを発見した。我々の評価は、インターネットで事前学習されたVLAモデルに挑戦するために、それらの先行研究で実施された評価よりもさらに大きな度合いの汎化をテストしていることに注意されたい。したがって、インターネットでの事前学習を持たないモデルのパフォーマンスが低くなることは予想される。RT-2-Xは明らかにRT-1-XとOctoの両方を上回っており、ロボティクスにとって大規模な事前学習済みVLMの利点を実証している。

注目すべきことに、OpenVLAは1桁小さい（550億パラメータに対して70億）にもかかわらず、Googleロボットの評価においてRT-2-Xと同等のパフォーマンスを示し、BridgeData V2の評価においてはRT-2-Xを大幅に上回っている。定性的に、我々はRT-2-XとOpenVLAの両方が、テストされた他のモデルと比べて顕著により堅牢な行動を示すことを発見した。例えば、気晴らしの物体が存在する場合に正しい物体に近づくこと、ターゲット物体の向きに合わせてロボットのエンドエフェクタの向きを適切に調整すること、さらには物体を不安定に掴むといったミスから回復することなどである（定性的なロールアウトの例については https://openvla.github.io を参照）。図3に示されるように、RT-2-Xは意味論的汎化のタスクにおいてより高いパフォーマンスを達成している。これは、RT-2-Xがより大規模なインターネット事前学習データを使用しており、OpenVLAのようにロボットデータのみでファインチューニングされるのではなく、事前学習の知識をより良く保持するためにロボットの行動データとインターネット事前学習データの両方で共ファインチューニング(co-fine-tuned)されていることを考えれば予想通りである。しかしながら、OpenVLAはBridgeData V2とGoogleロボットの両方の評価において、他のすべてのタスクカテゴリーで同等かそれ以上のパフォーマンスを示している。このパフォーマンスの違いは複数の要因の組み合わせに起因すると考えられる：我々はOpenVLAのためにより大規模な学習データセットである97万件の軌跡をキュレーションしたこと（RT-2-Xは35万件）。また、学習データセットのより慎重なクリーニングを実施し、例えばBridgeデータセットからすべてゼロの行動をフィルタリングしたこと（詳細な議論については付録Cを参照）。そして、OpenVLAが事前学習された意味論的および空間的特徴を組み合わせた融合視覚エンコーダを使用していることである。これらのコンポーネントのアブレーション分析については付録Dを参照されたい。

#### 5.2 新しいロボットセットアップへのデータ効率の良い適応 (Data-Efficient Adaptation to New Robot Setups)
これまでの研究は主にVLAの「そのままの状態 (out-of-the-box)」での直接評価 [^1], [^7], [^16] に焦点を当ててきたが、新しいタスクやロボットセットアップへのVLAモデルの効果的なファインチューニングはほとんど探求されておらず、それにもかかわらずVLAの普及にとって鍵となる。このセクションでは、新しい実世界のロボットセットアップに素早く適応するOpenVLAの能力を調査する。（シミュレーションにおけるファインチューニング実験については付録Eを参照）

**ロボットのセットアップとタスク (Robot setups and tasks).** 我々はOpenVLAモデルのためのシンプルなファインチューニングのレシピをテストする。それは、ターゲットタスクの10〜150回のデモンストレーションからなる小規模なデータセットを使用した、すべてのモデルパラメータの完全なファインチューニングである（図5を参照；パラメータ効率の良いファインチューニングのアプローチについてはセクション5.3で探求する）。我々はOpenVLAを2つのセットアップでテストする：Franka-Tabletop（テーブルの上に固定されたFranka Emika Panda 7自由度ロボットアーム）と、Franka-DROID（最近公開されたDROIDデータセット [^11] からのFrankaロボットアームのセットアップで、移動式のスタンディングデスクにマウントされている）である。これらのセットアップはそれぞれ5Hzと15Hzのノンブロッキングコントローラーを使用する。我々がファインチューニング実験のターゲットとなる身体としてFrankaロボットアームを選ぶ理由は、それらがロボット学習コミュニティで広く使用されており、したがってOpenVLAのファインチューニングの「ターゲット」となる可能性が高いからである。我々は様々なユースケースへのOpenVLAの適用性をテストするために、異なる制御周波数を持つセットアップでテストを行う。

図5：新しいロボットセットアップへの適応。我々は、7つのFranka Emika Pandaのタスク（各10〜150回のデモンストレーション）でゼロから学習された最先端のDiffusion Policyと、同じデータでファインチューニングされた汎用ロボット方策であるOctoおよびOpenVLAを評価する。Diffusion Policyは狭い単一指示のタスクで強力なパフォーマンスを示す一方で、OctoとOpenVLAは複数の指示と気晴らしの物体を含む多様なファインチューニングのタスクでより良いパフォーマンスを示す。全体として、OpenVLAは両方のセットアップにわたり最高の集計パフォーマンスを達成しており、下流タスクの方策を学習するための効果的なデフォルトになり得ることを示唆している。平均成功率±標準誤差(StdErr)は、各アプローチにつき129のロールアウト（Franka-Tabletopタスクでは99、Franka-DROIDタスクでは30）にわたって計算されている。詳細な結果については表7を参照。

**比較 (Comparisons).** 我々はゼロから学習された最先端のデータ効率の良い模倣学習アプローチであるDiffusion Policy [^3] と比較を行う。我々はまた、Diffusion Policyの入力と出力の仕様をOpenVLAに合わせたバージョンであるDiffusion Policy (matched) とも比較する[^3注]。さらに、ファインチューニングをサポートする最高の汎用方策として現在知られているOcto [^5] を、ターゲットのデータセットでファインチューニングして評価する（RT-2-Xのファインチューニングはその推論APIを通じてはサポートされていない）。また、我々はOpenVLAを同じターゲットのデータセット上でファインチューニングし、得られた方策をOpenVLAと表記する。最後に、アブレーション実験として、大規模なロボットの事前学習の利点を評価するために、OpenXで事前学習されたOpenVLAモデルをファインチューニングするのではなく、ターゲットのロボットセットアップ上で基礎となるベースのPrismatic VLMを直接ファインチューニングしたOpenVLA (scratch) と比較する。

[^3注]: 完全なDiffusion Policyは画像と固有受容状態(proprioceptive state)の両方を用いた2ステップの観測履歴を使用し、 `$T$` 個の将来の行動のチャンクを予測し、最初の `$X$` 個の行動をオープンループ方式で実行してから次のチャンクを予測することによって、後退ホライズン制御(receding horizon control)を実行する（15Hz制御の場合、DROIDの先行研究 [^11] と同様に `$T = 16, X = 8$` と設定する；5Hz制御の場合、チャンクサイズを `$T = 8, X = 3$` に減らす）。また、セクション5.2の中で絶対直交座標を予測してロボットを制御する唯一の手法である。他のすべての手法は相対位置制御を使用する。Diffusion Policy (matched) は単一の画像を入力として使用し、固有受容情報を持たず観測履歴も持たず、行動チャンクなしに単一の相対位置制御行動を予測する。

我々は結果を図5に提示する（タスクごとの内訳は付録の表7を参照）。我々は、両方のバージョンのDiffusion Policyが「ボウルにニンジンを入れる (Put Carrot in Bowl)」や「鍋にトウモロコシを注ぐ (Pour Corn into Pot)」のような狭い単一指示のタスクにおいては汎用方策であるOctoおよびOpenVLAと競争力があるか、それを上回ることを発見した。しかし、事前学習済みの汎用方策は、シーン内に複数の物体が存在し言語条件付けを必要とする、より多様なファインチューニングタスクにおいてより良いパフォーマンスを示す。OctoおよびOpenVLAにおけるOpenXの事前学習は、モデルが言語グラウンディングが重要となるこれらのより多様なタスクにより良く適応することを可能にする。我々はこの証拠をOpenVLA (scratch) のより低いパフォーマンスの中に確認している。

全体として、我々はOpenVLAが最も高い平均パフォーマンスを達成することを発見した。注目すべきことに、先行研究の多くは、狭い単一指示のタスクか多様なマルチ指示のタスクのいずれかにおいてのみ強力なパフォーマンスを達成しており、結果として成功率に大きなばらつきが生じている。OpenVLAはテストされたすべてのタスクにおいて少なくとも50%の成功率を達成した唯一のアプローチであり、特に多様な言語指示のセットを伴う場合、模倣学習タスクのための強力なデフォルトの選択肢となり得ることを示唆している。より狭く、しかし高い器用さを要求されるタスクについては、Diffusion Policyは依然としてより滑らかで正確な軌跡を示す。Diffusion Policyで実装されているように、行動のチャンキングや時間的平滑化を組み込むことで、OpenVLAが同レベルの器用さを達成する助けとなる可能性があり、将来の研究の有望な方向性となるかもしれない（現在の制限についての詳細な議論はセクション6を参照）。

#### 5.3 パラメータ効率の良いファインチューニング (Parameter-Efficient Fine-Tuning)
前のセクションにおけるOpenVLAの完全なファインチューニングの実行は、高いパフォーマンスを達成するために、タスクごとに5〜15時間（データセットのサイズに依存する）、8つのA100 GPUを使用した。これはVLAの事前学習に必要な計算量よりも大幅に少ないが、このセクションではさらに計算量およびパラメータ効率の良いファインチューニングアプローチを探求し、それらの有効性を調査する。

具体的には、我々は以下のファインチューニングアプローチを比較する：完全なファインチューニング(full fine-tuning)は、セクション5.2で説明したように、ファインチューニング中にすべての重みを更新する。最終層のみ(last layer only)は、OpenVLAのTransformerバックボーンの最後の層とトークン埋め込み行列のみをファインチューニングする。凍結された視覚(frozen vision)は視覚エンコーダを凍結するが他のすべての重みをファインチューニングする。サンドイッチファインチューニング(sandwich)は、視覚エンコーダ、トークン埋め込み行列、および最後の層の凍結を解除する。そしてLoRAは、Huら [^26] の人気のある低ランク適応技術をモデルのすべての線形層に適用し、複数のランク値 `$r$` を使用する。我々は、複数のFranka-Tabletopタスクにおけるファインチューニングの成功率と、学習パラメータの数およびGPUメモリの要件を、表1に報告する[^4注]。我々は、ネットワークの最後の層のみをファインチューニングしたり視覚エンコーダを凍結したりすることは低いパフォーマンスにつながることを発見し、ターゲットのシーンへの視覚的特徴のさらなる適応が不可欠であることを示唆している。対照的に、「サンドイッチファインチューニング」は視覚エンコーダをファインチューニングするためより良いパフォーマンスを達成し、完全なLLMバックボーンをファインチューニングしないためGPUメモリの消費量が少ない。最後に、LoRAはパフォーマンスと学習メモリの消費量の間で最良のトレードオフを達成しており、「サンドイッチファインチューニング」を上回り、パラメータのわずか1.4%をファインチューニングするだけで完全なファインチューニングのパフォーマンスに匹敵する。我々はLoRAのランクが方策のパフォーマンスに与える影響はごくわずかであることを発見したため、デフォルトのランクとして `$r = 32$` を使用することを推奨する。LoRAを用いれば、単一のA100 GPU上で10〜15時間以内にOpenVLAを新しいタスクにファインチューニングすることができる。これは完全なファインチューニングと比較して計算量が8分の1に削減されている。

[^4注]: セクション5.3およびセクション5.4において、我々はより小さなロボットデータ混合物（Octoと同じOpenXデータセットの混合物）で事前学習され、融合されたDinoSigLIPエンコーダの代わりにSigLIP [^79] の視覚バックボーンのみを使用する、わずかに小さなアーキテクチャを持つバージョンのOpenVLAモデルを使用して実験を行う。我々は、このよりシンプルなアーキテクチャがファインチューニングのタスクと「そのままの状態」のタスクの両方において依然として強力なパフォーマンスを達成することを発見した。

表1：パラメータ効率の良いファインチューニングの評価。LoRAファインチューニングはパフォーマンスと計算量の最良のトレードオフを達成し、モデルのパラメータのわずか1.4%を学習するだけで完全なファインチューニングのパフォーマンスに匹敵する。平均成功率±標準誤差(StdErr)は、選ばれたFranka-Tabletopタスクにおける各アプローチにつき33のロールアウトにわたって計算されている（詳細は表8を参照）。*: FSDP [^77] で2つのGPUに分割されている。

| Strategy | Success Rate | Train Params ($\times 10^6$) | VRAM (batch 16) |
| --- | --- | --- | --- |
| Full FT | 69.7 ± 7.2 % | 7,188.1 | 163.3 GB* |
| Last layer only | 30.3 ± 6.1 % | 465.1 | 51.4 GB |
| Frozen vision | 47.0 ± 6.9 % | 6,760.4 | 156.2 GB* |
| Sandwich | 62.1 ± 7.9 % | 914.2 | 64.0 GB |
| LoRA, rank=32 | 68.2 ± 7.5% | 97.6 | 59.7 GB |
| rank=64 | 68.2 ± 7.8% | 195.2 | 60.5 GB |

#### 5.4 量子化によるメモリ効率の良い推論 (Memory-Efficient Inference via Quantization)
パラメータ数70億のモデルであるOpenVLAは、推論時において、パラメータ数が `$< 100M$` のOctoのような先行するオープンソースの汎用方策よりも多くのメモリを消費する。我々はLLMのサービングからのベストプラクティスに従い、推論のためにOpenVLAをbfloat16精度で保存およびロードする（我々のデフォルトのアプローチ）。これによりメモリのフットプリントが半分に削減され、わずか16GBのGPUメモリを持つGPU上でOpenVLAをサービングできるようになる。このセクションでは、LLMのサービングのために開発された最新の量子化技術 [^27], [^88] を使用することで、方策の推論に必要なメモリをさらに削減し、VLA方策のアクセスしやすさを広げることができるかどうかをテストする。これらのアプローチはネットワークの重みをより低い精度でロードするため、メモリ要件の削減と、推論のスピードおよび精度の潜在的な低下とをトレードオフにする。

具体的には、我々は8つの代表的なBridgeData V2タスクにおいて、8-bitおよび4-bit精度でOpenVLAモデルをサービングすることを調査する。我々はメモリのフットプリントとロールアウトのパフォーマンスを表2に報告する。また、様々なコンシューマー向けおよびサーバー向けGPUで達成可能な制御周波数を図6に報告する。我々は、8-bit量子化が追加された量子化演算のオーバーヘッドのために、ほとんどのGPUにわたって推論を遅くすることを観察した。4-bit推論は、GPUメモリ転送の削減が量子化のオーバーヘッドを補うため、より高いスループットを達成する。推論速度の低下の結果として、8-bit量子化では大幅なパフォーマンスの低下が観察される。我々が評価に使用するA5000 GPU上では、モデルは1.2Hzでしか実行できず、BridgeData V2のタスクで使用される5Hzノンブロッキングコントローラーの学習データセットと比較して、システムダイナミクスが大幅に変化してしまう[^5注]。注目すべきことに、4-bit量子化はGPUメモリの半分未満の量しか必要としないにもかかわらず、bfloat16の半精度推論と同様のパフォーマンスをもたらす。4-bitで量子化されたモデルはA5000上で3Hzで実行できるため、データ収集時のシステムダイナミクスにより近く適合する。

[^5注]: 我々は、8-bitおよび4-bit量子化の両方が、学習データ上でオフラインで評価された場合にbfloat16推論と同等のトークン精度を達成するため、このパフォーマンスの低下は推論速度の低さに起因すると考える。裏付けとなる詳細については付録D.4を参照されたい。

図6：様々なGPUでのOpenVLAの推論速度。bfloat16およびint4量子化の両方が、特にAda LovelaceアーキテクチャのGPU (RTX 4090, H100) において高いスループットを達成する。TensorRT-LLM [^89] のような最新のLLM推論フレームワークによりさらなるスピードアップが可能である。♠: モデルは収めるために2つのGPUに分割されている。

表2：量子化された推論でのパフォーマンス。4-bit量子化は、GPUのメモリフットプリントを半分未満に削減しながら、bfloat16推論（我々のデフォルトアプローチ）のパフォーマンスに匹敵する。平均成功率±標準誤差(StdErr)は、8つの代表的なBridgeData V2のタスク [^6] および各アプローチにつき80のロールアウトにわたって計算されている（詳細は表5を参照）。

| Precision | Bridge Success | VRAM |
| --- | --- | --- |
| bfloat16 | 71.3 ± 4.8% | 16.8 GB |
| int8 | 58.1 ± 5.1% | 10.2 GB |
| int4 | 71.9 ± 4.7% | 7.0 GB |

### 6 議論と制限 (Discussion and Limitations)
本研究において、我々はそのままの状態で複数の身体にわたるロボット制御において強力なパフォーマンスを得る、最新のオープンソースの視覚-言語-行動モデルであるOpenVLAを提示した。我々はまた、パラメータ効率の良いファインチューニング技術を介して、OpenVLAが新しいロボットのセットアップに容易に適応できることも実証した。

現在のOpenVLAモデルにはいくつかの制限がある。まず、現在は単一の画像観測しかサポートしていない。現実には、実世界のロボットセットアップは異種多様であり、幅広い感覚入力の可能性がある [^5]。OpenVLAを拡張して、複数の画像や固有受容入力、および観測の履歴をサポートすることは、将来の研究に向けた重要な手段である。インターリーブされた画像とテキストデータで事前学習されたVLMの使用を探求することは、そのような柔軟な入力を持つVLAのファインチューニングを容易にするかもしれない。第二に、OpenVLAの推論スループットを向上させることは、50Hzで動作するALOHA [^90] のような高頻度な制御セットアップのためのVLA制御を可能にするために重要である。これにより、本研究で調査したタスクよりもさらに器用な、両手でのマニピュレーションタスクにおいてVLAをテストすることも可能になる。行動チャンキングや、投機的デコード [^91] のような代替の推論時最適化技術の使用を探求することは、潜在的な解決策を提供する。さらに、パフォーマンス向上の余地がある。OpenVLAは先行する汎用方策を上回っているものの、テストされたタスクにおいてまだ非常に高い信頼性を提供するには至っておらず、通常は `$< 90\%$` の成功率を達成している。最後に、計算量の制限のため、多くのVLAの設計に関する疑問が十分に探求されないままである。ベースとなるVLMのサイズはVLAのパフォーマンスにどのような影響を与えるか？ロボットの行動予測データとインターネット規模の視覚-言語データでの共学習(co-training)はVLAのパフォーマンスを大幅に向上させるか？VLAモデルに最も適した視覚的特徴は何か？我々は、OpenVLAモデルとコードベースの公開が、コミュニティによるこれらの疑問の共同調査を可能にすることを期待している。

---

### References

[^1]: Open X-Embodiment Collaboration, A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan, A. Raffin, A. Wahid, B. Burgess-Limerick, B. Kim, B. Schölkopf, B. Ichter, C. Lu, C. Xu, C. Finn, C. Xu, C. Chi, C. Huang, C. Chan, C. Pan, C. Fu, C. Devin, D. Driess, D. Pathak, D. Shah, D. Büchler, D. Kalashnikov, D. Sadigh, E. Johns, F. Ceola, F. Xia, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Schiavi, H. Su, H.-S. Fang, H. Shi, H. B. Amor, H. I. Christensen, H. Furuta, H. Walke, H. Fang, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Kim, J. Schneider, J. Hsu, J. Bohg, J. Bingham, J. Wu, J. Wu, J. Luo, J. Gu, J. Tan, J. Oh, J. Malik, J. Tompson, J. Yang, J. J. Lim, J. Silvério, J. Han, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Zhang, K. Majd, K. Rana, K. Srinivasan, L. Y. Chen, L. Pinto, L. Tan, L. Ott, L. Lee, M. Tomizuka, M. Du, M. Ahn, M. Zhang, M. Ding, M. K. Srirama, M. Sharma, M. J. Kim, N. Kanazawa, N. Hansen, N. Heess, N. J. Joshi, N. Suenderhauf, N. D. Palo, N. M. M. Shafiullah, O. Mees, O. Kroemer, P. R. Sanketi, P. Wohlhart, P. Xu, P. Sermanet, P. Sundaresan, Q. Vuong, R. Rafailov, R. Tian, R. Doshi, R. Martín-Martín, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kirmani, S. Levine, S. Moore, S. Bahl, S. Dass, S. Song, S. Xu, S. Haldar, S. Adebola, S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Dasari, S. Belkhale, T. Osa, T. Harada, T. Matsushima, T. Xiao, T. Yu, T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Darrell, V. Jain, V. Vanhoucke, W. Zhan, W. Zhou, W. Burgard, X. Chen, X. Wang, X. Zhu, X. Li, Y. Lu, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Xu, Y. Wang, Y. Bisk, Y. Cho, Y. Lee, Y. Cui, Y. hua Wu, Y. Tang, Y. Zhu, Y. Li, Y. Iwasawa, Y. Matsuo, Z. Xu, and Z. J. Cui. Open X-Embodiment: Robotic learning datasets and RT-X models. https://arxiv.org/abs/2310.08864, 2023.
[^2]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-1: Robotics transformer for real-world control at scale. In arXiv preprint arXiv:2212.06817, 2022.
[^3]: C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.
[^4]: A. Xie, L. Lee, T. Xiao, and C. Finn. Decomposing the generalization gap in imitation learning for visual robotic manipulation. arXiv preprint arXiv:2307.03659, 2023.
[^5]: Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. Tan, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy. https://octo-models.github.io, 2023.
[^6]: H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine. Bridgedata v2: A dataset for robot learning at scale, 2023.
[^7]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023.
[^8]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), volume 139, pages 8748–8763, 2021.
[^9]: X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In International Conference on Computer Vision (ICCV), 2023.
[^10]: H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
[^11]: A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, J. Gao, D. A. Herrera, M. Heo, K. Hsu, J. Hu, D. Jackson, C. Le, Y. Li, K. Lin, R. Lin, Z. Ma, A. Maddukuri, S. Mirchandani, D. Morton, T. Nguyen, A. O’Neill, R. Scalise, D. Seale, V. Son, S. Tian, E. Tran, A. E. Wang, Y. Wu, A. Xie, J. Yang, P. Yin, Y. Zhang, O. Bastani, G. Berseth, J. Bohg, K. Goldberg, A. Gupta, A. Gupta, D. Jayaraman, J. J. Lim, J. Malik, R. Martín-Martín, S. Ramamoorthy, D. Sadigh, S. Song, J. Wu, M. C. Yip, Y. Zhu, T. Kollar, S. Levine, and C. Finn. Droid: A large-scale in-the-wild robot manipulation dataset. 2024.
[^12]: S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022.
[^13]: S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang. Language-driven representation learning for robotics. ArXiv, abs/2302.12766, 2023. URL https://api.semanticscholar.org/CorpusID:257205716.
[^14]: M. Shridhar, L. Manuelli, and D. Fox. Cliport: What and where pathways for robotic manipulation. In Conference on robot learning, pages 894–906. PMLR, 2022.
[^15]: A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, B. Zitkovich, F. Xia, C. Finn, et al. Open-world object manipulation using pre-trained vision-language models. arXiv preprint arXiv:2303.00905, 2023.
[^16]: D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
[^17]: A. S. et al. Introducing rfm-1: Giving robots human-like reasoning capabilities, 2024. URL https://covariant.ai/insights/introducing-rfm-1-giving-robots-human-like-reasoning-capabilities/.
[^18]: Wayve. Lingo-2: Driving with natural language. 2024. URL https://wayve.ai/thinking/lingo-2-driving-with-language/.
[^19]: X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. M. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, A. Kolesnikov, J. Puigcerver, N. Ding, K. Rong, H. Akbari, G. Mishra, L. Xue, A. V. Thapliyal, J. Bradbury, W. Kuo, M. Seyedhosseini, C. Jia, B. K. Ayan, C. Riquelme, A. Steiner, A. Angelova, X. Zhai, N. Houlsby, and R. Soricut. Pali: A jointly-scaled multilingual language-image model. ArXiv, abs/2209.06794, 2022. URL https://api.semanticscholar.org/CorpusID:252222320.
[^20]: X. Chen, X. Wang, L. Beyer, A. Kolesnikov, J. Wu, P. Voigtlaender, B. Mustafa, S. Goodman, I. M. Alabdulmohsin, P. Padlewski, D. M. Salz, X. Xiong, D. Vlasic, F. Pavetic, K. Rong, T. Yu, D. Keysers, X.-Q. Zhai, and R. Soricut. PaLI-3 vision language models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023.
[^21]: T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, and ... Transformers: State-of-the-art natural language processing. In Proceedings of the 6th International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/1910.03771.
[^22]: H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
[^23]: A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
[^24]: G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
[^25]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
[^26]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
[^27]: T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36, 2024.
[^28]: Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Computer Vision and Pattern Recognition (CVPR), 2017.
[^29]: D. A. Hudson and C. D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Computer Vision and Pattern Recognition (CVPR), 2019.
[^30]: A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards VQA models that can read. In Computer Vision and Pattern Recognition (CVPR), 2019.
[^31]: J. P. Bigham, C. Jayant, H. Ji, G. Little, A. Miller, R. C. Miller, R. Miller, A. Tatarowicz, B. White, S. White, and T. Yeh. VizWiz: nearly real-time answers to visual questions. In User Interface Software and Technology (UIST), pages 333–342, 2010.
[^32]: S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, 2014.
[^33]: L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg. Modeling context in referring expressions. In European Conference on Computer Vision (ECCV), 2016.
[^34]: T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, P. Tafti, L. Hussenot, P. G. Sessa, A. Chowdhery, A. Roberts, A. Barua, A. Botev, A. Castro-Ros, A. Slone, A. Héliou, A. Tacchetti, A. Bulanova, A. Paterson, B. Tsai, B. Shahriari, C. L. Lan, C. A. Choquette-Choo, C. Crepy, D. Cer, D. Ippolito, D. Reid, E. Buchatskaya, E. Ni, E. Noland, G. Yan, G. Tucker, G.-C. Muraru, G. Rozhdestvenskiy, H. Michalewski, I. Tenney, I. Grishchenko, J. Austin, J. Keeling, J. Labanowski, J.-B. Lespiau, J. Stanway, J. Brennan, J. Chen, J. Ferret, J. Chiu, J. Mao-Jones, K. Lee, K. Yu, K. Millican, L. L. Sjoesund, L. Lee, L. Dixon, M. Reid, M. Mikuła, M. Wirth, M. Sharman, N. Chinaev, N. Thain, O. Bachem, O. Chang, O. Wahltinez, P. Bailey, P. Michel, P. Yotov, R. Chaabouni, R. Comanescu, R. Jana, R. Anil, R. McIlroy, R. Liu, R. Mullins, S. L. Smith, S. Borgeaud, S. Girgin, S. Douglas, S. Pandya, S. Shakeri, S. De, T. Klimenko, T. Hennigan, V. Feinberg, W. Stokowiec, Y. hui Chen, Z. Ahmed, Z. Gong, T. Warkentin, L. Peran, M. Giang, C. Farabet, O. Vinyals, J. Dean, K. Kavukcuoglu, D. Hassabis, Z. Ghahramani, D. Eck, J. Barral, F. Pereira, E. Collins, A. Joulin, N. Fiedel, E. Senter, A. Andreev, and K. Kenealy. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
[^35]: Y. Li, S. Bubeck, R. Eldan, A. D. Giorno, S. Gunasekar, and Y. T. Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.
[^36]: J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
[^37]: J. Li, D. Li, C. Xiong, and S. C. H. Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning (ICML), 2022.
[^38]: J. Li, D. Li, S. Savarese, and S. C. H. Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML), 2023.
[^39]: W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. A. Li, P. Fung, and S. C. H. Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
[^40]: H. H. Tan and M. Bansal. LXMERT: Learning cross-modality encoder representations from transformers. In Empirical Methods in Natural Language Processing (EMNLP), 2019.
[^41]: H. Laurençon, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. M. Rush, D. Kiela, M. Cord, and V. Sanh. OBELICS: An open web-scale filtered dataset of interleaved image-text documents. In Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks), 2023.
[^42]: H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
[^43]: H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
[^44]: S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865, 2024.
[^45]: D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke, et al. QT-Opt: Scalable deep reinforcement learning for vision-based robotic manipulation. arXiv preprint arXiv:1806.10293, 2018.
[^46]: D. Kalashnkov, J. Varley, Y. Chebotar, B. Swanson, R. Jonschkowski, C. Finn, S. Levine, and K. Hausman. Mt-opt: Continuous multi-task robotic reinforcement learning at scale. arXiv, 2021.
[^47]: F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021.
[^48]: K. Ehsani, T. Gupta, R. Hendrix, J. Salvador, L. Weihs, K.-H. Zeng, K. P. Singh, Y. Kim, W. Han, A. Herrasti, et al. Imitating shortest paths in simulation enables effective navigation and manipulation in the real world. arXiv preprint arXiv:2312.02976, 2023.
[^49]: H. Bharadhwaj, J. Vakil, M. Sharma, A. Gupta, S. Tulsiani, and V. Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. arXiv preprint arXiv:2309.01918, 2023.
[^50]: L. Pinto and A. Gupta. Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours. In 2016 IEEE international conference on robotics and automation (ICRA), pages 3406–3413. IEEE, 2016.
[^51]: A. Mandlekar, Y. Zhu, A. Garg, J. Booher, M. Spero, A. Tung, J. Gao, J. Emmons, A. Gupta, E. Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879–893. PMLR, 2018.
[^52]: A. Gupta, A. Murali, D. P. Gandhi, and L. Pinto. Robot learning in homes: Improving generalization and reducing dataset bias. Advances in neural information processing systems, 31, 2018.
[^53]: S. Dasari, F. Ebert, S. Tian, S. Nair, B. Bucher, K. Schmeckpeper, S. Singh, S. Levine, and C. Finn. Robonet: Large-scale multi-robot learning. CoRL, 2019.
[^54]: S. Cabi, S. G. Colmenarejo, A. Novikov, K. Konyushkova, S. Reed, R. Jeong, K. Zolna, Y. Aytar, D. Budden, M. Vecerik, O. Sushkov, D. Barker, J. Scholz, M. Denil, N. de Freitas, and Z. Wang. Scaling data-driven robotics with reward sketching and batch reinforcement learning. RSS, 2019.
[^55]: E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022.
[^56]: H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition@ CoRL2023, 3:5, 2023.
[^57]: C. Devin, A. Gupta, T. Darrell, P. Abbeel, and S. Levine. Learning modular neural network policies for multi-task and multi-robot transfer. In Proceedings of IEEE International Conference on Robotics and Automation, 2017.
[^58]: E. S. Hu, K. Huang, O. Rybkin, and D. Jayaraman. Know thyself: Transferable visual control policies through robot-awareness. In International Conference on Learning Representations, 2022.
[^59]: J. H. Yang, D. Sadigh, and C. Finn. Polybot: Training one policy across robots while embracing variability. In 7th Annual Conference on Robot Learning, 2023. URL https://openreview.net/forum?id=HEIRj51lcS.
[^60]: S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-maron, M. Giménez, Y. Sulsky, J. Kay, J. T. Springenberg, T. Eccles, J. Bruce, A. Razavi, A. Edwards, N. Heess, Y. Chen, R. Hadsell, O. Vinyals, M. Bordbar, and N. de Freitas. A generalist agent. Transactions on Machine Learning Research, 2022. ISSN 2835-8856.
[^61]: G. Salhotra, I.-C. A. Liu, and G. Sukhatme. Bridging action space mismatch in learning from demonstrations. arXiv preprint arXiv:2304.03833, 2023.
[^62]: I. Radosavovic, B. Shi, L. Fu, K. Goldberg, T. Darrell, and J. Malik. Robot learning with sensorimotor pre-training. In Conference on Robot Learning, 2023.
[^63]: D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine. Gnm: A general navigation model to drive any robot. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 7226–7233. IEEE, 2023.
[^64]: K. Bousmalis, G. Vezzani, D. Rao, C. Devin, A. X. Lee, M. Bauza, T. Davchev, Y. Zhou, A. Gupta, A. Raju, et al. Robocat: A self-improving foundation agent for robotic manipulation. arXiv preprint arXiv:2306.11706, 2023.
[^65]: D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine. ViNT: A foundation model for visual navigation. In 7th Annual Conference on Robot Learning, 2023. URL https://arxiv.org/abs/2306.14846.
[^66]: J. Yang, C. Glossop, A. Bhorkar, D. Shah, Q. Vuong, C. Finn, D. Sadigh, and S. Levine. Pushing the limits of cross-embodiment learning for manipulation and navigation. arXiv preprint arXiv:2402.19432, 2024.
[^67]: S. Y. Gadre, M. Wortsman, G. Ilharco, L. Schmidt, and S. Song. Cows on pasture: Baselines and benchmarks for language-driven zero-shot object navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23171–23181, 2023.
[^68]: Y. Du, K. Konyushkova, M. Denil, A. Raju, J. Landon, F. Hill, N. de Freitas, and S. Cabi. Vision-language models as success detectors. arXiv preprint arXiv:2303.07280, 2023.
[^69]: Y. J. Ma, V. Kumar, A. Zhang, O. Bastani, and D. Jayaraman. Liv: Language-image representations and rewards for robotic control. In International Conference on Machine Learning, pages 23301–23320. PMLR, 2023.
[^70]: X. Zhang, Y. Ding, S. Amiri, H. Yang, A. Kaminski, C. Esselink, and S. Zhang. Grounding classical task planners via vision-language models. arXiv preprint arXiv:2304.08587, 2023.
[^71]: S. Sontakke, J. Zhang, S. Arnold, K. Pertsch, E. Bıyık, D. Sadigh, C. Finn, and L. Itti. Roboclip: One demonstration is enough to learn robot policies. Advances in Neural Information Processing Systems, 36, 2024.
[^72]: J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang. An embodied generalist agent in 3d world. In Proceedings of the International Conference on Machine Learning (ICML), 2024.
[^73]: X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.
[^74]: H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan. 3d-vla: 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.
[^75]: PyTorch. Automatic mixed precision. URL https://pytorch.org/docs/stable/amp.html.
[^76]: T. Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.
[^77]: Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.
[^78]: N. Dorka, C. Huang, T. Welschehold, and W. Burgard. What matters in employing vision language models for tokenizing actions in robot control? In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024.
[^79]: X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.
[^80]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
[^81]: P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.
[^82]: C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
[^83]: O. Sidorov, R. Hu, M. Rohrbach, and A. Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020.
[^84]: H. Face. Introducing idefics: An open reproduction of state-of-the-art visual langage model. Hugging Face Blog, 2024.
[^85]: H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
[^86]: B. McKinzie, Z. Gan, J.-P. Fauconnier, S. Dodge, B. Zhang, P. Dufter, D. Shah, X. Du, F. Peng, F. Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.
[^87]: J. Lin, H. Yin, W. Ping, Y. Lu, P. Molchanov, A. Tao, H. Mao, J. Kautz, M. Shoeybi, and S. Han. Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533, 2023.
[^88]: T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems, 35: 30318–30332, 2022.
[^89]: NVIDIA. Tensorrt-llm. URL https://github.com/NVIDIA/TensorRT-LLM.
[^90]: T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.
[^91]: Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.
[^92]: A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.
[^93]: E. Rosete-Beas, O. Mees, G. Kalweit, J. Boedecker, and W. Burgard. Latent plans for task agnostic offline reinforcement learning. In Proceedings of the 6th Conference on Robot Learning (CoRL), 2022.
[^94]: O. Mees, J. Borja-Diaz, and W. Burgard. Grounding language with visual affordances over unstructured data. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023.
[^95]: S. Dass, J. Yapeter, J. Zhang, J. Zhang, K. Pertsch, S. Nikolaidis, and J. J. Lim. CLVR jaco play dataset, 2023. URL https://github.com/clvrai/clvr_jaco_play_dataset.
[^96]: J. Luo, C. Xu, X. Geng, G. Feng, K. Fang, L. Tan, S. Schaal, and S. Levine. Multi-stage cable routing through hierarchical imitation learning. arXiv preprint arXiv:2307.08927, 2023.
[^97]: A. Mandlekar, Y. Zhu, A. Garg, J. Booher, M. Spero, A. Tung, J. Gao, J. Emmons, A. Gupta, E. Orbay, S. Savarese, and L. Fei-Fei. RoboTurk: A crowdsourcing platform for robotic skill learning through imitation. CoRR, abs/1811.02790, 2018. URL http://arxiv.org/abs/1811.02790.
[^98]: Y. Zhu, A. Joshi, P. Stone, and Y. Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors, 2023.
[^99]: L. Y. Chen, S. Adebola, and K. Goldberg. Berkeley UR5 demonstration dataset. https://sites.google.com/view/berkeley-ur5/home.
[^100]: G. Zhou, V. Dean, M. K. Srirama, A. Rajeswaran, J. Pari, K. Hatch, A. Jain, T. Yu, P. Abbeel, L. Pinto, C. Finn, and A. Gupta. Train offline, test online: A real robot learning benchmark, 2023.
[^101]: C. Lynch, A. Wahid, J. Tompson, T. Ding, J. Betker, R. Baruch, T. Armstrong, and P. Florence. Interactive language: Talking to robots in real time. IEEE Robotics and Automation Letters, 2023.
[^102]: S. Belkhale, Y. Cui, and D. Sadigh. Hydra: Hybrid robot actions for imitation learning. arxiv, 2023.
[^103]: Y. Zhu, P. Stone, and Y. Zhu. Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation. IEEE Robotics and Automation Letters, 7(2):4126–4133, 2022.
[^104]: Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto. From play to policy: Conditional behavior generation from uncurated robot data. arXiv preprint arXiv:2210.10047, 2022.
[^105]: M. Heo, Y. Lee, D. Lee, and J. J. Lim. Furniturebench: Reproducible real-world benchmark for long-horizon complex manipulation. In Robotics: Science and Systems, 2023.
[^106]: G. Yan, K. Wu, and X. Wang. ucsd kitchens Dataset. August 2023.
[^107]: S. Nasiriany, T. Gao, A. Mandlekar, and Y. Zhu. Learning and retrieval from prior data for skill-based imitation learning. In Conference on Robot Learning (CoRL), 2022.
[^108]: H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu. Robot learning on the job: Human-in-the-loop autonomy and learning during deployment. In Robotics: Science and Systems (RSS), 2023.
[^109]: G. Quere, A. Hagengruber, M. Iskandar, S. Bustamante, D. Leidner, F. Stulp, and J. Vogel. Shared Control Templates for Assistive Robotics. In 2020 IEEE International Conference on Robotics and Automation (ICRA), page 7, Paris, France, 2020.
[^110]: S. Saxena, M. Sharma, and O. Kroemer. Multi-resolution sensing for real-time control with vision-language models. In 7th Annual Conference on Robot Learning, 2023. URL https://openreview.net/forum?id=WuBv9-IGDUA.
[^111]: R. Shah, R. Martín-Martín, and Y. Zhu. MUTEX: Learning unified policies from multimodal task specifications. In 7th Annual Conference on Robot Learning, 2023. URL https://openreview.net/forum?id=PwqiqaaEzJ.
[^112]: X. Zhu, R. Tian, C. Xu, M. Ding, W. Zhan, and M. Tomizuka. Fanuc manipulation: A dataset for learning-based manipulation with fanuc mate 200id robot. 2023.
[^113]: R. Mendonca, S. Bahl, and D. Pathak. Structured world models from human videos. CoRL, 2023.
[^114]: J. Luo, C. Xu, F. Liu, L. Tan, Z. Lin, J. Wu, P. Abbeel, and S. Levine. Fmb: a functional manipulation benchmark for generalizable robotic learning. arXiv preprint arXiv:2401.08553, 2024.
[^115]: N. M. M. Shafiullah, A. Rai, H. Etukuru, Y. Liu, I. Misra, S. Chintala, and L. Pinto. On bringing robots home, 2023.
[^116]: B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36, 2024.
[^117]: V. Sanh, L. Debut, J. Chaumond, and T. Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

### A データ混合の詳細 (Data Mixture Details)
我々の使用したデータ混合を表3に示す。この混合は大部分が [^5] に従っており、いくつかの追加データセットが含まれている。

表3: Open X-Embodimentデータセット [^1] からのデータセットを使用したOpenVLAの学習データ混合。[^5] に従い、いくつかを追加している。

| OpenVLA Training Dataset Mixture | |
| :--- | :--- |
| Fractal [^92] | 12.7% |
| Kuka [^45] | 12.7% |
| Bridge[^6], [^47] | 13.3% |
| Taco Play [^93], [^94] | 3.0% |
| Jaco Play [^95] | 0.4% |
| Berkeley Cable Routing [^96] | 0.2% |
| Roboturk [^97] | 2.3% |
| Viola [^98] | 0.9% |
| Berkeley Autolab UR5 [^99] | 1.2% |
| Toto [^100] | 2.0% |
| Language Table [^101] | 4.4% |
| Stanford Hydra Dataset [^102] | 4.4% |
| Austin Buds Dataset [^103] | 0.2% |
| NYU Franka Play Dataset [^104] | 0.8% |
| Furniture Bench Dataset [^105] | 2.4% |
| UCSD Kitchen Dataset [^106] | <0.1% |
| Austin Sailor Dataset [^107] | 2.2% |
| Austin Sirius Dataset [^108] | 1.7% |
| DLR EDAN Shared Control [^109] | <0.1% |
| IAMLab CMU Pickup Insert [^110] | 0.9% |
| UTAustin Mutex [^111] | 2.2% |
| Berkeley Fanuc Manipulation [^112] | 0.7% |
| CMU Stretch [^113] | 0.2% |
| BC-Z [^55] | 7.5% |
| FMB Dataset [^114] | 7.1% |
| DobbE [^115] | 1.4% |
| DROID [^11] | 10.0%[^6注] |

[^6注]: 学習の進捗が遅いため（セクション3.3を参照）、学習の最後の3分の1についてはDROIDを削除し、その混合の重みを他のすべてのデータセットに再分配した。

---

### B 評価タスクと詳細な結果 (Evaluation Tasks and Detailed Results)
本セクションでは、セクション5.1で議論したBridgeData V2 WidowXおよびGoogleロボットの評価、ならびにセクション5.2で議論したFranka-TabletopおよびFranka-DROIDのファインチューニング評価に関する詳細を提供する。

#### B.1 BridgeData V2 WidowX 評価の詳細 (BridgeData V2 WidowX Evaluation Details)
ここでは、セクション5.1で議論したBridgeData V2の評価に特に焦点を当てる。

**B.1.1 BridgeData V2 評価タスク (BridgeData V2 Evaluation Tasks)**
セクション5.1で説明したように、我々は各汎用ロボットマニピュレーション方策を17のタスクで各10回の試行により評価する。本セクションでは、タスクのカテゴリーと個々のタスクに関する詳細を提供する。

合計で、我々は5つの視覚的汎化タスク、2つの動作的汎化タスク、3つの物理的汎化タスク、4つの意味論的汎化タスク、および3つの言語グラウンディングタスクについて評価を行う。元のデータセットで使用されたのと全く同じ物体を調達することができないため、我々が評価するすべてのタスクは何らかの形の分布シフトを導入していることに注意されたい（元々別の場所で構築された実世界のテスト環境を再現するため、他の分布シフトも自然に発生する。このような分布シフトの詳細な議論については付録B.1.2を参照）。全17タスクは図7に描かれている。各ロールアウトは失敗（0）または成功（1）としてマークされる。いくつかのより困難なタスクでは、部分的な成功（0.5）を記録する。部分的なクレジットの条件については、以下のタスク説明で記述する。以下では、図7に示される順序で17のタスクのそれぞれについて説明する：

1. **ボウルにナスを入れる (Easy Version) (Put Eggplant into Pot (Easy Version)):** ロボットの目標はナスを拾い上げて鍋に落とすことである。（元の鍋を調達できなかったため）元のBridgeData V2学習データセットで使用された鍋とは外観が異なる手作りの紙の鍋を使用するため、これは視覚的汎化タスクである。他の16のタスクすべてとは異なり、この特定のタスクについては、方策をロールアウトする前にロボットのエンドエフェクタをナスの真上に初期化する。そのため、これを「ボウルにナスを入れる」タスクの「Easy Version」と呼ぶ。
2. **ボウルにナスを入れる (Put Eggplant into Pot):** このタスクは、ロボットのエンドエフェクタがナスの真上に初期化されないことを除いて、上記で説明したものと同じである。代わりに、すべてのロールアウトにわたって固定された位置に初期化する。つまり、ロボットはナスを操作する前に、まず水平に手を伸ばしてナスに到達しなければならない。（注：以下で説明する他のすべてのタスクにも同じことが適用される。）上記と同じ理由により、これは視覚的汎化タスクである。
3. **カウンターからシンクにカップを入れる (Put Cup from Counter into Sink):** ロボットの目標は、キッチンのカウンターまたは水切りラックからピンクのカップを拾い上げ、右側のシンクに入れることである。青いカップではなくピンクのカップを使用するため、これは視覚的汎化タスクである（元のBridgeData V2データセットでは青いカップが使用されているが、我々が評価した手法はいずれもそれを確実に操作することができなかった。おそらく、カップの色がシンクの色と溶け込んでしまうためである）。
4. **鍋にナスを入れる（気晴らしあり） (Put Eggplant into Pot (w/ Clutter)):** このタスクは、いくつかの気晴らしの物体が存在するためにより難しいことを除いて、「鍋にナスを入れる」タスクと同じである。通常の「鍋にナスを入れる」タスクで議論したのと同じ理由により、さらにシーン内に未経験の気晴らしがあることを考慮すると、これは視覚的汎化タスクである。ロボットが正しいターゲット物体に向かって移動した場合、部分的なクレジット（1のうち0.5）が与えられる。
5. **ピンクの皿に黄色いトウモロコシを置く (Put Yellow Corn on Pink Plate):** ロボットの目標は、黄色いトウモロコシを拾い上げ、ピンクの皿の上に置くことである。シンクの奥のセクションのカウンターにある緑色の恐竜のような、シーン内の未経験の気晴らしの物体の存在により、これは視覚的汎化タスクである。ロボットが正しいターゲット物体に向かって移動した場合、部分的なクレジット（1のうち0.5）が与えられる。
6. **ナスを持ち上げる (Lift Eggplant):** ロボットの目標は、ナスを掴んで空中に持ち上げることである。ナスが未経験の位置および/または向きに初期化されており、ロボットがタスクを完了するためには学習分布の位置および/または向きを超えて移動し、しばしば遠距離へのリーチを実行せざるを得ないため、これは動作的汎化タスクである。（注：元のBridgeData V2のデモンストレーションでは、この環境において遠距離へのリーチは示されていない。詳細は付録B.1.2を参照）。一見シンプルに見えるこのタスクが、多くの方策にとって驚くほど難しいことがわかった。ロボットがナスに接触した場合、部分的なクレジット（1のうち0.5）が与えられる。
7. **皿にニンジンを置く（高さ変更あり） (Put Carrot on Plate (w/ Height Change)):** ロボットの目標は、ニンジンを拾い上げて黄色い皿の上に置くことである。皿がシンクの底の通常の位置から高くされており、ロボットが（その過程で皿を倒すことなく）高くされたプラットフォーム上のニンジンを正しく置くために軌道を調整しなければならないため、これは動作的汎化タスクである。ロボットがニンジンを掴み、それで皿に触れた場合、部分的なクレジット（1のうち0.5）が与えられる。
8. **皿にニンジンを置く (Put Carrot on Plate):** このタスクは、皿が通常の位置（シンクまたは水切りラックの底）にあることを除いて、上記と同じである。元のBridgeData V2データセットで使用されたニンジンとはサイズと形状が異なり、より短く狭いため、我々はこれを物理的汎化タスクとみなす。（注：同じニンジンを使用しているため、上記のこのタスクの以前のバージョンも技術的には物理的汎化タスクであるが、そこでの焦点が動作的汎化であるため、「動作的汎化」のカテゴリーの下にリストしている）。
9. **鍋を直立にひっくり返す (Flip Pot Upright):** ロボットの目標は、エピソードの終わりにシンク内で鍋が直立の向きになるように鍋を操作することである。この鍋は、元のBridgeData V2の学習デモンストレーションで使用されたものとはサイズと形状が異なる（我々が使用する鍋はより広く、より短い）ため、これは物理的汎化タスクである。
10. **単4電池を持ち上げる (Lift AAA Battery):** ロボットの目標は、単4電池を掴んで空中に持ち上げることだけである。この電池は、この環境におけるBridgeData V2学習デモンストレーションで見られるターゲット物体よりもはるかに小さく薄いため、これは物理的汎化タスクとみなされる。詳細は付録B.1.2を参照。（注：このターゲット物体はこの環境における元のBridgeData V2のデモンストレーションには存在しないため、これは「意味論的汎化」の例でもあるが、ここでの主な焦点が物理的汎化であるため、「物理的汎化」としてのみ分類している）。
11. **ドクロを水切りラックに移動させる (Move Skull into Drying Rack):** ロボットの目標は、ドクロのぜんまいのおもちゃを掴み、シンクの左側にある黄色の水切りラックに落とすことである。ドクロは未経験のターゲット物体（BridgeData V2の学習デモンストレーションには現れない）であるため、これは意味論的汎化タスクである。
12. **白いテープを持ち上げる (Lift White Tape):** ロボットの目標は、白いテープのロールを掴み、空中に持ち上げることである。白いテープのロールは未経験のターゲット物体（BridgeData V2の学習デモンストレーションには現れない）であるため、これは意味論的汎化タスクである。（注：この環境における学習デモンストレーションで見られる物体とは形状が異なるため、このタスクは「物理的汎化」とみなすこともできる。多くの方策はこのリング構造を持つ物体を掴むのに苦労し、しばしばロボットのエンドエフェクタを直接中央の領域に移動させてしまう）。
13. **鍋から紫色のブドウを取り出す (Take Purple Grapes out of Pot):** ロボットの目標は、スチール製の鍋の中にある紫色のブドウを掴み、（それを持ち上げたり、鍋の外のどこかに落としたりして）鍋から取り出すことである。これは未経験の言語指示であるため、意味論的汎化タスクである。ロボットは元のBridgeData V2学習データセットにおいてこのタスクを見たことがない。
14. **青いカップをピンクのカップの上に重ねる (Stack Blue Cup on Pink Cup):** ロボットの目標は、青いカップを掴み、ピンクのカップの上にしっかりと置くことである。これは未経験の言語指示であるため、意味論的汎化タスクである。ロボットは元のBridgeData V2学習データセットにおいて、この環境でのこのタスクを見たことがない。ロボットが青いカップを掴み、青いカップでピンクのカップに触れた場合、部分的なクレジット（1のうち0.5）が与えられる。
15. **{ナス, 赤いボトル} を鍋に入れる (Put {Eggplant, Red Bottle} into Pot):** これは言語グラウンディングタスクである。ロボットの目標は、指定されたターゲット物体を鍋に入れることである。ナスと赤いボトルの両方がシーン内に存在する。我々はペアでの評価を実施する：同じ初期状態について、あるエピソードではナスをターゲットにするように方策にプロンプトを与え、次のエピソードでは赤いボトルをターゲットにするようにプロンプトを与える。我々は、両方のターゲット物体について同じ5つの初期状態のセットを使用し、各手法をナスで5回、赤いボトルで5回テストする。ロボットが正しいターゲット物体に向かって移動した場合、部分的なクレジット（1のうち0.5）が与えられる。
16. **{チーズ, 赤い唐辛子} を持ち上げる (Lift {Cheese, Red Chili Pepper}):** これは言語グラウンディングタスクである。ロボットの目標は、指定されたターゲット物体を掴んで持ち上げることである。上のタスクで説明したように、ペアでの評価を実施する。ロボットが正しいターゲット物体に向かって移動した場合、部分的なクレジット（1のうち0.5）が与えられる。
17. **{青いカップ, ピンクのカップ} を皿に置く (Put {Blue Cup, Pink Cup} on Plate):** これは言語グラウンディングタスクである。ロボットの目標は、指定されたターゲット物体を掴み、皿の上に置くことである。他の言語グラウンディングタスクで説明したように、ペアでの評価を実施する。ロボットが正しいターゲット物体に向かって移動した場合、部分的なクレジット（1のうち0.5）が与えられる。

図7: BridgeData V2 WidowXロボットの評価タスク。我々は、すべての汎用ロボット方策を4つのタイプの分布外 (OOD) 汎化タスク（セクション5.1で定義された、視覚、動作、物理、および意味論）について評価する。画像の各ペアは、開始状態と、ロボットがタスクを完了した後の終了状態の例を示している。我々はまた、初期状態を固定したままプロンプトを変更し、方策が正しいターゲット物体に接近できるかどうかをテストすることにより、下の3行に示された3つのタスクにおいて言語グラウンディングを厳密に評価する。

**B.1.2 評価タスクと元のBridgeData V2学習データとの比較 (Comparing Evaluation Tasks to Original BridgeData V2 Training Data)**
我々は、元のBridgeData V2データセット [^6] で使用されたシンク環境で評価を実施する。我々は、シンクに対するロボットの位置やシーンに対するカメラの配置を大まかに近似することで、BridgeData V2データセットの元の環境と一致するように環境を再現する。元のデータセットにはこれらの位置の正確な測定値がないため、正確な環境のセットアップを再現することはできず、ロボット、シンク、カメラの配置がわずかに異なることによる自然な分布シフトが生じる。さらに、学習デモンストレーションが収集された場所とは異なる場所でロボットの方策を評価するため、他の自然な分布シフトも生じる。例えば、照明条件や背景（例：シンクの後ろの見える領域）は、学習データセットで見られたものとは必然的に異なる。さらに、元のBridgeData V2データセットで使用されたのと全く同じ物体のセットを調達することができないため、学習時に使用された物体とテスト時に使用された物体の間には分布シフトがある。

これらすべての課題にもかかわらず、OpenVLAやRT-2-Xのようないくつかの汎用方策は、依然として汎化し、「そのままの状態」でさまざまなタスクをかなり確実に実行できることがわかった。RT-1-XやOctoのような他の汎用方策もいくつかのタスクを完了することができるが、BridgeData V2の評価スイートにおけるより困難な汎化タスクでテストされた場合には苦戦する。

元のBridgeData V2データセットには、この特定のシンク環境での以下の7つのタスクのデモンストレーションが含まれている：「鍋を直立にひっくり返す (Flip Pot Upright)」、「皿にニンジンを置く (Put Carrot on Plate)」、「カウンター（または水切りラック）からシンクにカップを入れる (Put Cup from Counter (or Drying Rack) into Sink)」、「鍋にナスを入れる (Put Eggplant into Pot)」、「まな板にナイフを置く (Put Knife on Cutting Board)」、「鍋にスプーンを入れる (Put Spoon in Pot)」、および「レバーを縦にして前に向ける (Turn Lever Vertical to Front)」。元のデータセットからのこれらすべてのタスクのサンプル画像については図8を参照されたい。この環境で収集されたすべての学習デモンストレーションは、エピソードの開始時にロボットのエンドエフェクタがターゲット物体の直上に位置するように初期化されていることに注意されたい。（ただし、これはBridgeData V2データセットのすべての環境に当てはまるわけではない。他のいくつかの環境では、ロボットはターゲット物体からさらに離れた位置に初期化されるため、操作する前にまず水平に物体に向かって手を伸ばさなければならない。）

図8: 元のBridgeData V2のシンク環境でのタスク。元のBridgeData V2データセットのシンク環境でのサンプルデモンストレーションの画像から、この環境でのすべてのデモンストレーションは、ロボットのエンドエフェクタがターゲット物体の直上に位置するように初期化されていたことがわかる。これらの初期状態は、図7に示すBridgeData V2の評価タスクで使用する初期状態とは異なることに注意されたい。我々の評価では、ターゲット物体の直上に配置するのではなく、常にシンクの上の固定された場所にロボットのエンドエフェクタを初期化する（1つのタスク「鍋にナスを入れる (Easy Version)」を除く）。

我々のBridgeData V2の評価スイートでは、「鍋にナスを入れる (Easy Version)」という1つのタスクのみが、ロボットのエンドエフェクタがターゲット物体の上を直接ホバリングしている状態で初期化される。他の16のタスクすべてにおいて、エンドエフェクタはシンクの上の固定された場所に初期化されるため、ロボットは物体に向かって水平に手を伸ばさなければならない。この初期条件は、我々の評価スイートにおけるさまざまなタイプのOOD汎化で導入される分布シフトと組み合わされて汎用方策に挑戦し、タスクを正常に完了するために高度な堅牢性を要求する。したがって、RT-1-XやOctoのような方策の成功率は、先行研究で報告されているものよりも低くなる。しかし、RT-2-XやOpenVLAのような他の方策は、これらすべての分布シフトと課題にもかかわらず、依然として比較的強力なパフォーマンスを達成することがわかる。

**B.1.3 BridgeData V2の評価結果の詳細 (Detailed BridgeData V2 Evaluation Results)**
BridgeData V2 WidowXの完全な評価結果については表4を参照されたい。17のタスクのそれぞれについて、10回の試行のうち各手法が成功した数が記載されている。OpenVLAは大多数のタスクで最も強力なパフォーマンスを達成し、汎用方策の中で最も高い集計成功率を示している。RT-2-Xも良好なパフォーマンスを示し、RT-1-XやOctoを上回っているが、OpenVLAほどのパフォーマンスは発揮していない。RT-1-XとOctoは、これらの汎化タスクにおいて概して困難を経験している。

表4: BridgeData V2 WidowXの詳細な評価結果。視覚/動作/物理/意味論的汎化タスクと言語グラウンディングタスクを含む、17タスクの完全な評価スイート（セクション5.1で議論）でのパフォーマンスを報告する。いくつかのタスクでは部分的な成功（スコア0.5）が可能であることに注意されたい。詳細は付録B.1.1を参照。OpenVLAがほとんどのタスクで最高のパフォーマンスを発揮し、全体として最高のパフォーマンスを達成し、それにRT-2-Xが続くことがわかった。一方で、RT-1-XとOctoは評価において苦戦し、いくつかのタスクで0〜2回の成功しか得られなかった。すべてのタスクの図については図7を参照されたい。

| Category | Task | # Trials | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Visual gen | Put Eggplant into Pot (Easy Version) | 10 | 1 | 5 | 7 | 10 |
| Visual gen | Put Eggplant into Pot | 10 | 0 | 1 | 5 | 10 |
| Visual gen | Put Cup from Counter into Sink | 10 | 1 | 1 | 0 | 7 |
| Visual gen | Put Eggplant into Pot (w/ Clutter) | 10 | 1 | 3.5 | 6 | 7.5 |
| Visual gen | Put Yellow Corn on Pink Plate | 10 | 1 | 4 | 8 | 9 |
| Motion gen | Lift Eggplant | 10 | 3 | 0.5 | 6.5 | 7.5 |
| Motion gen | Put Carrot on Plate (w/ Height Change) | 10 | 2 | 1 | 4.5 | 4.5 |
| Physical gen | Put Carrot on Plate | 10 | 1 | 0 | 1 | 8 |
| Physical gen | Flip Pot Upright | 10 | 2 | 6 | 5 | 8 |
| Physical gen | Lift AAA Battery | 10 | 0 | 0 | 2 | 7 |
| Semantic gen | Move Skull into Drying Rack | 10 | 1 | 0 | 5 | 5 |
| Semantic gen | Lift White Tape | 10 | 3 | 0 | 0 | 1 |
| Semantic gen | Take Purple Grapes out of Pot | 10 | 6 | 0 | 5 | 4 |
| Semantic gen | Stack Blue Cup on Pink Cup | 10 | 0.5 | 0 | 5.5 | 4.5 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot | 10 | 2.5 | 4 | 8.5 | 7.5 |
| Language grounding | Lift {Cheese, Red Chili Pepper} | 10 | 1.5 | 2.5 | 8.5 | 10 |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate | 10 | 5 | 5.5 | 8.5 | 9.5 |
| | Mean Success Rate | | 18.5±2.7% | 20.0±2.6% | 50.6±3.5% | 70.6±3.2% |

さらに、表5では、表2に要約された量子化された推論実験の完全な評価結果を提供する。これらの評価では、完全な評価スイート内のすべてのタスクカテゴリーにまたがる8つの代表的なBridgeData V2タスクで方策をテストする。

表5: 完全な量子化された推論結果。ここでは、表2に示された結果の詳細版を提示する。

| Category | Task | # Trials | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Visual gen | Put Eggplant into Pot (Easy Version) | 10 | 9 | 7 | 9 |
| Visual gen | Put Eggplant into Pot | 10 | 7 | 7 | 7 |
| Visual gen | Put Cup from Counter into Sink | 10 | 5 | 3 | 7 |
| Motion gen | Lift Eggplant | 10 | 6 | 4 | 7.5 |
| Physical gen | Put Carrot on Plate | 10 | 6 | 5 | 7 |
| Physical gen | Lift AAA Battery | 10 | 7 | 5 | 3 |
| Semantic gen | Take Purple Grapes out of Pot | 10 | 8 | 8 | 9 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot | 10 | 9 | 7.5 | 8 |
| | Mean Success Rate | | 71.3 ± 4.8% | 58.1 ± 5.1% | 71.9 ± 4.7% |

#### B.2 Googleロボットの評価の詳細 (Google Robot Evaluation Details)
本セクションでは、セクション5.1で紹介したGoogleロボットの評価に関する詳細を提供する。

**B.2.1 Googleロボットの評価タスク (Google Robot Evaluation Tasks)**
Googleロボットにおいて、我々は各汎用ロボット方策を12のタスクでそれぞれ5回のロールアウト、合計60回のロールアウトで評価する。最初の5つのタスクは分布内条件でのテストであり、最後の7つのタスクはより難しい分布外 (OOD) 条件でのテストである。すべてのタスクは図9に描かれている。各ロールアウトは失敗（0）または成功（1）としてマークされる。
以下に12のタスクを説明する：

1. **コーラの缶を拾う (分布内) (Pick Coke Can (in-distribution)):** ロボットは、上にコーラの缶が置かれたプラットフォームの前に配置される。ロボットの目標は、コーラの缶を掴んで持ち上げることである。
2. **緑色の缶の近くにリンゴを移動させる (分布内) (Move Apple near Green Can (in-distribution)):** ロボットは、上にリンゴと緑色のソーダの缶が置かれたプラットフォームの前に配置される。ロボットの目標は、リンゴを掴み、それを緑色の缶の隣に移動させることである。
3. **青いポテトチップスの袋をリンゴの近くに移動させる (分布内) (Move Blue Chip Bag near Apple (in-distribution)):** ロボットは、上に青いポテトチップスの袋とリンゴが置かれたプラットフォームの前に配置される。ロボットの目標は、青いポテトチップスの袋を掴み、リンゴの近くに移動させることである。
4. **コーラの缶を直立させる (分布内) (Place Coke Can Upright (in-distribution)):** ロボットは、上にコーラの缶が置かれたプラットフォームの前に配置され、缶は横向きに水平に置かれている。ロボットの目標は、コーラの缶を掴み、垂直になるように向きを変えることである。
5. **真ん中の引き出しを開ける (分布内) (Open Middle Drawer (in-distribution)):** ロボットは、3つの引き出しのセットの前に配置される。ロボットの目標は、真ん中の引き出しの取っ手を掴み、引き出しを引いて開けることである。
6. **オレンジを茶色いポテトチップスの袋の近くに移動させる (OOD) (Move Orange near Brown Chip Bag (OOD)):** ロボットは、上に茶色いポテトチップスの袋とオレンジが置かれたプラットフォームの前に配置される。青空と白い雲の模様のテーブルクロスが、物体の下にあるプラットフォームを覆っている。ロボットの目標は、オレンジを掴み、ポテトチップスの袋の隣に持っていくことである。オレンジが学習データセットに対して未経験の物体であり、テーブルクロスが未経験の背景であるため、このタスクはOODである。[^7]
[^7]: Googleロボットの評価におけるOOD条件の詳細なリストについては、Brohanら [^7] の付録を参照されたい。
7. **ペプシの缶を拾う (OOD) (Pick Pepsi Can (OOD)):** ロボットは、上にペプシの缶が置かれたプラットフォームの前に配置される。明るい黄色/茶色の模様のテーブルクロスが、缶の下にあるプラットフォームを覆っている。ロボットの目標は、缶を掴んで持ち上げることである。ペプシの缶が未経験の物体であり、テーブルクロスが未経験の背景であるため、このタスクはOODである。
8. **バナナを拾う (OOD) (Pick Banana (OOD)):** ロボットは、リンゴ、コーラの缶、バナナが置かれたプラットフォームの前に配置される。ロボットの目標は、バナナを掴んで持ち上げることである。バナナが未経験のターゲット物体であるため、このタスクはOODである。
9. **緑色のカップを拾う (OOD) (Pick Green Cup (OOD)):** ロボットは、バナナ、ペプシの缶、緑色のカップが置かれたプラットフォームの前に配置される。ロボットの目標は、緑色のカップを掴んで持ち上げることである。シーン内のすべての物体が学習データにおいて未経験であるため、このタスクはOODである。
10. **皿にリンゴを置く (OOD) (Place Apple on Plate (OOD)):** ロボットは、皿とリンゴが置かれたプラットフォームの前に配置される。ロボットの目標は、リンゴを掴み、皿の上に移動させることである。未経験の物体関係を記述した新しい指示であるため、このタスクはOODである：学習デモンストレーションは、リンゴを皿の「上に置く」のではなく、皿の「近くに移動させる」ことのみをカバーしている。
11. **鍋にバナナを入れる (OOD) (Place Banana in Pan (OOD)):** ロボットは、鍋とバナナが置かれたプラットフォームの前に配置される。ロボットの目標は、バナナを掴み、鍋の中に移動させることである。前のタスクで説明したように、バナナが未経験のターゲット物体であり、かつ未経験の物体関係を記述した新しい指示であるため、このタスクはOODである。
12. **コーラの缶をテイラー・スウィフトのところに移動させる (OOD) (Move Coke Can to Taylor Swift (OOD)):** ロボットは、コーラの缶とテイラー・スウィフトを含む3人の異なる有名人の写真が置かれたプラットフォームの前に配置される。ロボットの目標は、缶を掴み、テイラー・スウィフトの写真のところに移動させることである。有名人の写真がロボットのインタラクションデータにおいて未経験であるため、このタスクはOODである。

図9: Googleロボットの評価タスク。我々は、すべての汎用ロボット方策を分布内タスクおよび分布外 (OOD) 汎化タスクについて評価する。OODタスクには、未経験の背景、ターゲット物体、指示/物体関係、および意味論的概念（例：ロボットの行動データには現れないインターネット上の写真）が含まれる。

**B.2.2 Googleロボットの評価結果の詳細 (Detailed Google Robot Evaluation Results)**

表6: Googleロボットの詳細な評価結果。セクション5.1で議論したGoogleロボットの評価の完全な評価結果を報告する。各汎用方策は、分布内および分布外 (OOD) のテスト条件の両方をカバーする12のタスクにわたる60のロールアウトで評価される。一番下の行では、各方策の平均成功率±標準誤差(StdErr)を報告する。OpenVLAとRT-2-Xは全体としてRT-1-XとOctoの両方を大きく上回っている（エラーバーが重なっているため、両方の平均成功率を太字にしている）。すべてのタスクの図については図9を参照されたい。

| Category | Task | # Trials | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| In-distribution | Pick Coke Can | 5 | 5 | 1 | 5 | 5 |
| In-distribution | Move Apple near Green Can | 5 | 3 | 3 | 3 | 5 |
| In-distribution | Move Blue Chip Bag near Apple | 5 | 0 | 3 | 4 | 5 |
| In-distribution | Place Coke Can Upright | 5 | 0 | 0 | 4 | 4 |
| In-distribution | Open Middle Drawer | 5 | 0 | 4 | 2 | 3 |
| OOD | Move Orange near Brown Chip Bag | 5 | 1 | 2 | 5 | 5 |
| OOD | Pick Pepsi Can | 5 | 3 | 0 | 5 | 4 |
| OOD | Pick Banana | 5 | 5 | 3 | 5 | 5 |
| OOD | Pick Green Cup | 5 | 1 | 0 | 5 | 5 |
| OOD | Place Apple on Plate | 5 | 0 | 0 | 4 | 4 |
| OOD | Place Banana in Pan | 5 | 0 | 0 | 2 | 4 |
| OOD | Move Coke Can near Taylor Swift | 5 | 2 | 0 | 3 | 2 |
| | Mean Success Rate | | 33.3±6.1% | 26.7±5.8% | **78.3±5.4%** | **85.0±4.6%** |

Googleロボットの評価の完全な結果は表6に示されている。全体として、RT-1-XとOctoは評価タスクで困難を経験しており、いくつかのタスクで5回の試行のうち1回も成功を達成できないことがよくあることがわかった。一方で、RT-2-XとOpenVLAは強力なパフォーマンスを実証し、5回の試行のうち少なくとも2回はすべてのタスクを完了している。これら2つのVLA方策は、この特定の評価スイートにおいて互いに同等のパフォーマンスを発揮する。

#### B.3 データ効率の良い適応実験の詳細 (Data-Efficient Adaptation Experiment Details)
本セクションでは、セクション5.2で議論したデータ効率の良い適応実験に関する詳細を提供する。ここでは、Franka-TabletopやFranka-DROIDのような新しいロボットセットアップにおけるファインチューニングされたOpenVLA方策の有効性を調査する。

**B.3.1 Franka-Tabletop および Franka-DROID タスク (Franka-Tabletop and Franka-DROID Tasks)**
我々は、7つのタスクのそれぞれについて10〜150回のデモンストレーションを収集する。最初の6つのタスクは、我々が「Franka-Tabletop」（テーブルの上に取り付けられたFranka Emika Pandaロボット）と呼ぶロボットセットアップに対応し、最後のタスクは我々が「Franka-DROID」と呼ぶロボットセットアップに対応する。
Franka-Tabletopセットアップにおいて、6つのタスクのうち最初の3つは単一指示タスクに対応しており狭いものであるが、最後の3つのタスクは、複数の物体がシーンに存在し、ロボットが言語指示に応じて正しいものを操作しなければならないマルチ指示タスクに対応している。

図10: Franka-Tabletop ファインチューニングタスク。セクション5.2のデータ効率の良い適応実験で使用され、図10で詳細に説明されているFranka-Tabletopタスクを上に描画している。上の3行に示される6つのタスクのうち最初の3つは単一の指示のみを伴うが、下の3行の最後の3つのタスクは複数の物体と指示（指示はターゲット物体またはターゲット位置を指定する）を伴う。最初の列は学習データ分布に一致するサンプルの初期状態を示し、2番目の列は分布外 (OOD) の初期状態（例：未経験の背景、ターゲット物体、気晴らし、物体の位置/向き）を示す。セクション5.2のすべての方策は、分布内タスクで10〜12回のロールアウト、OODタスクで5〜6回のロールアウトで評価される。

以下に、図10に示す6つのFranka-Tabletopタスクについて説明する：

1. **ボウルにニンジンを入れる (単一指示) (Put Carrot in Bowl (single-instruction)):** ロボットの目標は、ニンジンを掴んでボウルに入れることである。このタスクについて50回のデモンストレーションを学習データセット用に収集し、各エピソードでテーブル上の異なる場所にニンジンとボウルをランダムに配置する。ニンジンは常にボウルの左側に初期化される。評価中、各試行は成功（1）または失敗（0）として記録され、部分的なクレジットはない。
2. **鍋にトウモロコシを注ぐ (単一指示) (Pour Corn into Pot (single-instruction)):** ロボットの目標は、赤いボウルを掴み、スチール製の鍋に向かって移動し、中身（黄色いトウモロコシ）を鍋に注ぐことである。このタスクについて50回のデモンストレーションを学習データセット用に収集し、各エピソードでテーブル上の異なる場所にボウルと鍋をランダムに配置する。ボウルは常に鍋の右側に初期化される。評価中、各試行は成功（1）または失敗（0）として記録され、部分的なクレジットはない。
3. **鍋を直立にひっくり返す (単一指示) (Flip Pot Upright (single-instruction)):** ロボットの目標は、スチール製の鍋（最初は縦向きに置かれている）を掴み、直立の位置になるように回転させ、テーブルの上に置き直すことである。学習データセット用にはこのタスクのデモンストレーションを10回のみ収集し、テーブルの小さなセクション内の様々な場所にスチール製の鍋をランダムに配置する。評価中、各試行は成功（1）、失敗（0）、または部分的な成功（0.5）として記録される。部分的な成功には、鍋を掴むが直立させない場合や、直立の位置まで倒すが慎重に誘導しない場合が含まれる。フルクレジットを得るためには、ロボットはエピソードの終わりに鍋を離さなければならない。
4. **`<object>` を皿に移動させる (マルチ指示) (Move `<object>` onto Plate (multi-instruction)):** ロボットの目標は、（言語指示で指定されたターゲットに応じて）3つの物体のうち1つを掴み、テーブルの右側にある皿の上に置くことである。このタスクについて150回のデモンストレーションを学習データセット用に収集し、テーブル上に3つの物体の異なる組み合わせをランダムに配置し、1つをターゲットとして選択する。皿は常にテーブルの右側に初期化される。評価中、各試行は成功（1）、失敗（0）、または部分的な成功（0.5）として記録される。ロボットが最初に接触した物体が正しいターゲット物体（つまり、言語指示で指定された物体）であるが、ロボットがタスクを完了しなかった場合に部分的な成功が記録される。
5. **`<object>` を倒す (マルチ指示) (Knock `<object>` Over (multi-instruction)):** ロボットの目標は、（言語指示で指定されたターゲットに応じて）3つの物体のうち1つに接近し、倒れるまでそれを押すことである。このタスクについて70回のデモンストレーションを学習データセット用に収集し、テーブル上に3つの物体の異なる組み合わせをランダムに配置し、1つをターゲットとして選択する。評価中、各試行は成功（1）、失敗（0）、または部分的な成功（0.5）として記録される。ロボットが最初に接触した物体が正しいターゲット物体（つまり、言語指示で指定された物体）であるが、ロボットがタスクを完了しなかった場合に部分的な成功が記録される。
6. **`<object>` をタオルで覆う (マルチ指示) (Cover `<object>` with Towel (multi-instruction)):** ロボットの目標は、青いタオルを掴み、（言語指示で指定されたターゲットに応じて）3つの物体のうち1つの上に置くことである。このタスクについて45回のデモンストレーションを学習データセット用に収集し、テーブル上に3つの物体の異なる組み合わせをランダムに配置する。評価中、各試行は成功（1）、失敗（0）、または部分的な成功（0.5）として記録される。ロボットがタオルで最初に触れた物体が正しいターゲット物体（つまり、言語指示で指定された物体）であるが、ロボットがタスクを完了しなかった場合（例えば、ターゲット物体の上ではなくテーブルの上にタオルを落としてしまった場合）に部分的な成功が記録される。タオルの任意の部分がターゲット物体の上面にある場合、すなわち物体が完全に覆われている必要はなく、フルクレジットが与えられる。

すべてのFranka-Tabletopタスクについて、我々は各手法を10〜12回の分布内試行と5〜6回のOOD汎化試行で評価する。分布内とOODのテスト条件は図10（2列目）に描かれている。
以下に、6つのタスクそれぞれのOODテスト条件について説明する：

1. **ボウルにニンジンを入れる (OOD) (Put Carrot in Bowl (OOD)):** ニンジンの代わりにナス（未経験の物体）が置かれる。
2. **鍋にトウモロコシを注ぐ (OOD) (Pour Corn into Pot (OOD)):** 未経験の茶色いテーブルクロスがテーブルトップを覆う。
3. **鍋を直立にひっくり返す (OOD) (Flip Pot Upright (OOD)):** 未経験の白いテーブルクロスがテーブルトップを覆う。
4. **`<object>` を皿に移動させる (OOD) (Move `<object>` onto Plate (OOD)):** 3つの未経験の物体のセットがテーブル上に置かれる。
5. **`<object>` を倒す (OOD) (Knock `<object>` Over (OOD)):** 3つの経験済みの物体のセットの後ろに、2つの未経験の気晴らしの物体（赤いプラスチックのカップと茶色い箱）が配置される。
6. **`<object>` をタオルで覆う (OOD) (Cover `<object>` with Towel (OOD)):** テーブル上の3つの物体が逆さまに、未経験の位置に置かれる。

最後に、Franka-DROID環境において、我々は1つのタスクとそのバリエーションを実験する：テーブルを拭く (Wipe Table)（図11を参照）。このタスクでは、ロボットの目標はブラシを掴み、3つの小さな茶色い物体をすべてちりとりの中に掃き入れることである。すべての物体の位置を変えながら、学習データセット用にこのタスクの70回のデモンストレーションを収集する。

図11: Franka-DROID ファインチューニングタスク。ここに示されている「テーブルを拭く (Wipe Table)」タスクは、セクション5.2のデータ効率の良い適応実験で使用される最後のタスクである。左の画像は分布内試行の初期条件を示している。右の画像は、テーブルの上に未経験の気晴らしの物体が存在する分布外試行を示している。タスクを完全に完了するには、ロボットはブラシを掴み、3つの物体をすべてちりとりの中に掃き入れなければならない。

テスト時には、学習データに一致する分布内条件（図11、左）に加えて、テーブル上のシーンに気晴らしの物体も存在する分布外 (OOD) 条件（図11、右）で評価する。各試行にはさまざまな結果が考えられるため、我々は次のような採点基準を定義する：各試行の最大スコアは2ポイントである。ロボットが3つの物体をすべてちりとりの中に掃き入れた場合、方策は満点の2ポイントを受け取る。1つまたは2つの物体をちりとりの中に掃き入れることに成功した場合は1ポイントを受け取る。それ以外の場合は0ポイントを受け取る。我々は各方策を18回の分布内試行と12回のOOD試行で評価するため、各方策は60ポイント満点の総スコアを受け取る。

**B.3.2 Franka-Tabletop および Franka-DROID の詳細な評価結果 (Detailed Franka-Tabletop and Franka-DROID Evaluation Results)**
Franka-TabletopおよびFranka-DROIDの完全な評価結果は表7に示されている。我々はセクション5.2で議論された手法を評価する。Diffusion Policyは単一指示のFranka-Tabletopタスク（例えば「ボウルにニンジンを入れる」や「鍋にトウモロコシを注ぐ」）で強力なパフォーマンスを実証し、他の手法を上回ることがわかった。しかし、より多様なマルチ指示タスク（「`<object>` を皿に移動させる」、「`<object>` を倒す」、および「`<object>` をタオルで覆う」）では、OpenVLAとOctoがより高いパフォーマンスを達成している。Franka-DROID環境においては、OpenVLAが最高の結果を得ている。全体として、OpenVLAは両方のタスクにわたって最高の平均パフォーマンスを達成していることがわかった。さらに、表8には、表1に要約されたパラメータ効率の良いファインチューニング実験の結果の詳細版を示す。これらの実験では、分布内とOODの両方のバリエーションを持つ2つのFranka-Tabletopタスクの代表的なサブセットを使用する：1つの狭い単一指示タスク（「ボウルにニンジンを入れる」）と1つの多様なマルチ指示タスク（「`<object>` を皿に移動させる」）。セクション5.2で使用されたのと同じ数（それぞれ50と150）の学習デモンストレーションを使用する。詳細は付録B.3.1で説明されている。

表7: データ効率の良い適応実験の詳細な結果。ここでは、図5に要約された結果の完全な内訳を提示する。新しいロボットタスクでゼロから学習されたDiffusion Policyと、同じデータでファインチューニングされた汎用方策のパフォーマンスを報告する。各方策は、分布内および分布外 (OOD) の両方の汎化条件に対してテストされる（Franka-Tabletopタスクについては図10、Franka-DROIDタスクについては図11を参照）。すべてのタスクで最高の結果を出す単一の方策はないことがわかった：Diffusion Policyは単一指示タスクで高い成功率を達成する一方で、OpenVLAとOctoは多様なマルチ指示タスクで優れたパフォーマンスを発揮する。しかし、集計されたパフォーマンスの観点からは、OpenVLAが両方の環境にわたって最高の平均成功率を獲得している。

| | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo | OpenVLA (scratch) | OpenVLA (ours) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Franka-Tabletop (5Hz)** | | | | | | |
| "Put Carrot in Bowl" (in-distribution) | 10 | 90.0% | 80.0% | 40.0% | 70.0% | 70.0% |
| "Put Carrot in Bowl" (OOD) | 5 | 20.0% | 0.0% | 20.0% | 0.0% | 40.0% |
| "Pour Corn into Pot" (in-distribution) | 10 | 100.0% | 90.0% | 0.0% | 10.0% | 50.0% |
| "Pour Corn into Pot" (OOD) | 5 | 80.0% | 60.0% | 0.0% | 20.0% | 60.0% |
| "Flip Pot Upright" (in-distribution) | 10 | 100.0% | 85.0% | 40.0% | 85.0% | 100.0% |
| "Flip Pot Upright" (OOD) | 5 | 50.0% | 20.0% | 0.0% | 40.0% | 80.0% |
| "Move `<object>` onto Plate" (in-distribution) | 12 | 25.0% | 25.0% | 41.7% | 8.3% | 75.0% |
| "Move `<object>` onto Plate" (OOD) | 6 | 8.3% | 33.3% | 8.3% | 33.3% | 58.3% |
| "Knock `<object>` Over" (in-distribution) | 12 | 33.3% | 25.0% | 83.3% | 75.0% | 75.0% |
| "Knock `<object>` Over" (OOD) | 6 | 16.7% | 16.7% | 33.3% | 58.3% | 83.3% |
| "Cover `<object>` with Towel" (in-distribution) | 12 | 16.7% | 20.8% | 91.7% | 41.7% | 50.0% |
| "Cover `<object>` with Towel" (OOD) | 6 | 16.7% | 33.3% | 91.7% | 50.0% | 50.0% |
| Average | | 48.5±4.9% | 43.4±4.7% | 43.4±4.4% | 43.4±4.6% | 67.2±4.0% |
| **Franka-DROID (15Hz)** | | | | | | |
| "Wipe Table" (in-distribution) | 18 | 50.0% | 27.8% | 52.8% | 25.0% | 55.6% |
| "Wipe Table" + Distractors (OOD) | 12 | 12.5% | 25.0% | 16.7% | 16.7% | 62.5% |
| Average | | 35.0±8.0% | 26.7±7.5% | 38.3±8.5% | 21.7±6.6% | 58.3±7.2% |

表8: パラメータ効率の良いファインチューニング実験の詳細な結果。ここでは、表1に要約された詳細なタスクパフォーマンス結果を提示する。

| | # trials | Full FT | Last layer only | Frozen vision | Sandwich | LoRA, r=32 | LoRA, r=64 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Franka-Tabletop (5Hz)** | | | | | | | |
| "Put Carrot in Bowl" (in-distribution) | 10 | 90.0 | 40.0 | 40.0 | 90.0 | 60.0 | 90.0 |
| "Put Carrot in Bowl" (OOD) | 5 | 40.0 | 0.0 | 40.0 | 0.0 | 60.0 | 40.0 |
| "Move `<object>` onto Plate" (in-distribution) | 12 | 79.2 | 33.3 | 50.0 | 75.0 | 75.0 | 62.5 |
| "Move `<object>` onto Plate" (OOD) | 6 | 41.7 | 33.3 | 58.3 | 41.7 | 75.0 | 66.7 |
| Average | | 69.7±7.2% | 30.3±6.1% | 47.0±6.9% | 62.1±7.9% | 68.2±7.5% | 68.2±7.8% |

### C RT-2-XとOpenVLAのBridgeData V2評価における比較 (RT-2-X vs. OpenVLA in BridgeData V2 Evaluations)
このセクションでは、セクション5.1で議論したBridgeData V2の評価におけるRT-2-XとOpenVLAの比較について追加の詳細を提供する。以前に議論したように、OpenVLAはRT-2-Xよりも大規模なOpenXデータのサブセットで事前学習されており、単一の視覚エンコーダではなく融合されたSigLIP-DinoV2視覚バックボーンを使用している。しかし、これらの要因に加えて、特にBridgeData V2の評価においてOpenVLAがRT-2-Xを大幅に上回ったこと（図3を参照）は、Bridgeデータセットのより慎重な前処理に起因すると我々は考えている。

OpenVLAモデルの開発中、我々はBridgeData V2データセットの元のバージョンには、すべてゼロ（何もしない）の行動を伴う多くの遷移(transition)が含まれていることを発見した。例えば、すべてのデモンストレーションにおいて、最初のタイムステップでグラウンドトゥルースの行動としてすべてゼロの行動が記録されていた。結果として、いかなるデータの前処理も行わずに元のデータセットで表現力の高いVLAモデルを学習させると、すべてゼロの行動を頻繁に予測し、評価中にフリーズしてしまう方策ができあがってしまった。そのため、OpenVLAモデルを学習する際には、すべてのデモンストレーションの最初の遷移を単にフィルタリングして除外し、ほとんどの場合においてフリーズする動作を軽減するにはこれで十分であった。

しかし、RT-2-Xモデルはそのようなデータの前処理なしで学習されているため、そのままの状態で展開され、モデルのクエリ手順を変更しない場合、前述のフリーズする動作を頻繁に引き起こし、ロールアウトのパフォーマンスを著しく低下させる。これはプロプライエタリなモデルであり、再学習することが不可能であるため（例えば、我々の前処理済みBridgeData V2データセットで）、我々はこの問題を軽減するために、最も確率の高い行動がしばしばすべてゼロであったのに対し、2番目に確率の高い行動はそうではなかったため、単にモデルから2番目に確率の高い行動をクエリした。（注：これは、Open X-Embodimentの実験 [^1] で報告されているBridgeData V2の評価において、RT-2-Xモデルの開発者らによって適用されたのと同じ回避策(workaround)である。）この回避策によりBridgeData V2の評価におけるRT-2-Xのパフォーマンスは大幅に向上したが、データセットの前処理済みバージョンでモデルを再学習する場合と比較すると依然として最適ではないと我々は考えている。

我々はまた、RT-2-Xを動的にクエリすることも試みた。つまり、最初に最も確率の高い行動をサンプリングし、それがすべてゼロであった場合にのみ2番目に確率の高い行動をサンプリングするという方法である。しかし、常に2番目に確率の高い行動をクエリするよりも、動的なクエリの方がパフォーマンスが低下することを経験的に発見した。我々はこれが、動的クエリから生じるロボットのダイナミクスの変化に起因すると仮説を立てている。モデルを再クエリするために軌道の途中で一時停止することは、クエリパイプラインにおける無視できないレイテンシのためにロボットの動きにわずかな中断をもたらし、これが微妙なパフォーマンスの低下につながる。したがって、我々はOpen X-Embodimentプロジェクト [^1] で行われたように、常に2番目に確率の高い行動をクエリした際のRT-2-Xのパフォーマンスを報告する。

---

### D 追加実験およびアブレーション (Additional Experiments and Ablations)
このセクションでは、OpenVLAモデルのアーキテクチャおよび学習スキームの個々のコンポーネントが与える影響を分析するためのいくつかの追加実験を行い、この研究の前のセクションで行われた主張に対する定量的な証拠を提供する。我々は以下の疑問に答えることを目指す：

1. OpenXの学習はどれほど重要であり、それはOpenVLAのパフォーマンスにどのように影響するか（付録D.1）？
2. 融合されたSigLIP-DinoV2視覚エンコーダを使用することは、SigLIPのみの視覚エンコーダを使用する場合と比較して、OpenVLAのパフォーマンスにどのような影響を与えるか（付録D.2）？
3. OpenVLAにおいて視覚エンコーダをファインチューニングするのと凍結するのとではどちらが良いか（付録D.3）？
4. セクション5.3で議論された量子化された推論の結果は、方策のパフォーマンスがモデルの推論速度から切り離された場合、どのように変化するか（付録D.4）？

上記のそれぞれの疑問に対処する実験のセットアップと結果について、以下のセクションで順に議論する。

#### D.1 OpenX 学習データのアブレーション実験 (OpenX Training Data Ablation Experiments)
セクション3.3で議論したように、OpenVLAはOpen X-Embodimentデータセット [^1] (OpenX) からのロボットの身体、シーン、およびタスクの大規模なデータセット上で学習されている。このセクションでは、OpenXの混合をアブレーションし、OpenXの学習が方策のパフォーマンスに与える影響を評価するために、1つのロボットのデータセットのみでVLA方策を学習させる。セクション5.2で議論したように（OpenVLA (scratch) を参照）、ファインチューニングの領域においてOpenXの学習をアブレーションすることの悪影響はすでに観察されているが、このセクションではさらなる裏付けとなる証拠を提供するために、別のロボットの身体での追加の実験について議論する。

**実験のセットアップとタスク (Experimental setup and tasks).**
我々は、元のOpenVLAモデルと、OpenVLAと同じ事前学習済みVLM（Prismatic VLM [^44]）を採用し、付録Aで議論した完全なOpenXの学習混合の代わりにBridgeData V2 [^6] のみでファインチューニングすることによって作成されたOpenVLA-Bridgeとを比較する。付録B.1.1で議論したBridgeData V2 WidowXロボットの評価スイートから、8つの代表的なタスクのサブセットでOpenVLAとOpenVLA-Bridgeを評価する。タスクは表9にリストされている。

**結果 (Results).**
OpenXの学習混合のアブレーションの結果は表9に示されている。OpenVLAとOpenVLA-Bridgeを比較すると、パフォーマンスが劇的に低下していることがわかり（絶対成功率で30%の減少）、これは最終的な方策のパフォーマンスに対するOpenXの事前学習の重要性を示している。言語グラウンディングのパフォーマンスには影響がないものの、すべての汎化カテゴリーにおいてパフォーマンスの低下が観察される。この結果は、OpenXの学習混合におけるシーン、物体、タスクの大きな多様性が、OpenVLAモデルの向上した汎化能力を解き放つために不可欠であることを示唆している。

表9: BridgeData V2 WidowXのアブレーション実験結果。OpenVLAモデルのアーキテクチャおよび学習スキームの様々なコンポーネントの重要性を評価するために、8つの代表的なタスクのサブセットにおいて様々な手法を評価する。OpenVLA-BridgeはOpenXの学習なしのOpenVLAのバージョンであり（BridgeData V2のみで学習されている）、OpenVLA-Bridge-SigLIPはさらにDinoV2エンコーダを削除することによって融合された視覚バックボーンをアブレーションしたものである（その視覚バックボーンはSigLIPエンコーダのみで構成されている）。OpenXの学習と融合された視覚エンコーダの両方が方策のパフォーマンスを向上させることが観察されるが、前者は後者よりもはるかに大きな効果を持っている。

| Category | Task | # Trials | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Visual gen | Put Eggplant into Pot (Easy Version) | 10 | 10 | 8 | 8 |
| Visual gen | Put Eggplant into Pot | 10 | 10 | 2 | 3 |
| Visual gen | Put Cup from Counter into Sink | 10 | 7 | 4 | 2 |
| Motion gen | Lift Eggplant | 10 | 7.5 | 5.5 | 6.5 |
| Physical gen | Put Carrot on Plate | 10 | 8 | 4 | 1 |
| Physical gen | Lift AAA Battery | 10 | 7 | 2 | 2 |
| Semantic gen | Take Purple Grapes out of Pot | 10 | 4 | 3 | 3 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot | 10 | 7.5 | 8 | 7 |
| | Mean Success Rate | | 76.3 ± 4.8% | 45.6 ± 5.6% | 40.6 ± 5.5% |

#### D.2 デュアル vs. シングル視覚エンコーダ実験 (Dual vs. Single Vision Encoder Experiments)
OpenVLAモデルのアーキテクチャは、SigLIP [^9] とDinoV2 [^25] のエンコーダを組み合わせた融合視覚バックボーンで構成されている。このセクションでは、デュアル視覚エンコーダを使用することの重要性を評価するためにDinoV2コンポーネントをアブレーションする。

**実験のセットアップとタスク (Experimental setup and tasks).**
我々は、BridgeData V2のみで学習され、視覚バックボーンとしてSigLIPエンコーダのみで構成されるOpenVLAのバージョンであるOpenVLA-Bridge-SigLIPというモデルをインスタンス化する。このモデルを、前のセクション（付録D.1）で議論したOpenVLA-Bridgeモデルと比較する。これは元のOpenVLAモデルと同じモデルアーキテクチャを共有し、Bridgeのロボットデータでのみ学習されている。したがって、OpenVLA-Bridge-SigLIPとOpenVLA-Bridgeの唯一の違いは、前者が視覚バックボーンにおいてDinoV2エンコーダを省略していることである。我々は前のセクションで説明したのと同じ8つのBridgeタスクのサブセットでこれらのモデルを評価する。

**結果 (Results).**
デュアル視覚エンコーダのアブレーションの結果は表9に示されている。OpenVLA-BridgeからOpenVLA-Bridge-SigLIPへのパフォーマンスの低下は、視覚バックボーンに追加でDinoV2エンコーダを含めることが方策のパフォーマンスを向上させることを示唆している。しかし、ここでの5%のパフォーマンスの低下は、OpenXの学習のアブレーションで観察された30%のパフォーマンスの低下ほどは大きくない。DinoV2で表現される低レベルの空間的特徴は、一部のケースでのみ汎化を助けるように思われる。

#### D.3 ファインチューニング vs. 凍結視覚エンコーダ実験 (Fine-Tuned vs. Frozen Vision Encoder Experiments)
セクション3.4で議論したように、VLMに関する先行研究では、視覚エンコーダのパラメータをファインチューニングするよりも、それを凍結した方が高いパフォーマンスが得られることが観察されている [^44]。しかし、OpenVLAの学習時においては、SigLIP-DinoV2視覚バックボーンを含め、モデル内のすべての70億のパラメータをファインチューニングした。なぜなら、開発の初期段階において、視覚エンコーダをファインチューニングすることがよりパフォーマンスの高いVLAにつながることを我々が発見したためである。これは様々な事前学習済みVLMおよびモデルアーキテクチャにわたって当てはまる発見であった。我々は以下にそのような発見の詳細を議論する。

**実験のセットアップとタスク (Experimental setup and tasks).**
このセクションでは、Prismatic VLMs [^44] リポジトリからの2つの異なる事前学習済みモデルをBridgeData V2上でファインチューニングすることによって作成された、2つのVLA方策のパフォーマンスを報告する。これら2つの事前学習済みモデルはSigLIP ViT-SO 224pxおよびLLaVa v1.5 7B (Reproduction)と名付けられている。それらのアーキテクチャおよび学習の混合に関する詳細についてはKaramchetiら [^44] を参照されたい。表10に示される様々なBridgeタスクで両方の方策を評価する。ここでの評価の構成は、以前に議論したBridgeの評価とは異なるため、結果は他の同様の実験の結果と直接比較することはできないことに注意されたい。

**結果 (Results).**
ファインチューニング vs. 凍結視覚エンコーダ実験の結果は表10に示されている。テストされた両方のVLAにおいて、視覚エンコーダをファインチューニングすることが様々なタスクにわたって大幅に高い成功率につながることがわかった。定性的に、一部のケースでは、凍結された視覚エンコーダの方策を展開すると、明らかに最適ではない不安定なロボットの行動につながる。結果として、我々は開発の初期段階で、凍結された視覚エンコーダを使用したさらなる実験を行わないことを決定した。

表10: ファインチューニング vs. 凍結視覚エンコーダ実験結果。Prismatic VLMs [^44] リポジトリからの2つの異なる事前学習済みVLMの上に構築された2つのVLA方策において、視覚エンコーダのファインチューニング（"Fine-Tuned"）と凍結（"Frozen Vision"）のパフォーマンスを評価する。ここに示されているBridgeData V2 WidowXのタスクは、本研究の他のBridge実験に使用されたのと同じシンク環境で実行される（ただし、これらの評価はプロジェクトの初期段階で実施されたため、ここでの初期環境の構成は異なる）。良好な方策のパフォーマンスを得るためには、視覚エンコーダのファインチューニングが重要であることがわかった。凍結視覚エンコーダを用いた一部の評価は、非常に低い（ゼロに近い）パフォーマンスと不安定なロボットの行動のために中止された。凍結視覚エンコーダとファインチューニングされたアプローチの両方がテストされた評価のうち、視覚エンコーダをファインチューニングすることで平均成功率80.0%を達成したのに対し、凍結したままにした場合は平均成功率46.7%であった。

| Task | # Trials | SigLIP ViT-SO 224px (Frozen Vision # Successes) | SigLIP ViT-SO 224px (Fine-Tuned # Successes) | LLaVa v1.5 7B (Frozen Vision # Successes) | LLaVa v1.5 7B (Fine-Tuned # Successes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Put Eggplant into Pot | 10 | 7 | 10 | 5 | 9 |
| Put Corn on Plate | 10 | 10 | 9 | 0 | 9 |
| **Mean Success Rate** | | 85 | 95 | 25 | 90 |
| Put { Eggplant, Red Bottle } into Pot | 4 | 2 | 4 | – | 3 |
| Put { Blue Cup, Pink Cup } on Plate | 4 | 0 | 0 | – | 0 |
| Lift { Cheese, Red Chili Pepper } | 4 | 0 | 3 | – | 2 |
| Put { Strawberry, Lime } into Pot | 4 | 1 | 0 | – | 3 |
| Move { Sushi, Grapes } | 4 | 3 | 4 | – | 3 |
| **Mean Success Rate** | | 30 | 55 | – | 55 |

#### D.4 追加の量子化推論実験：方策のパフォーマンスとモデルの推論速度の切り離し (Additional Quantized Inference Experiments: Disentangling Policy Performance and Model Inference Speed)
セクション5.3において、我々は推論時に異なる精度レベル（半精度(bfloat16)、8-bit量子化、および4-bit量子化）でOpenVLAを評価した。8-bit量子化は他の2つのアプローチと比較してBridgeData V2のパフォーマンスを低下させており、我々はこのパフォーマンスの低下は8-bit量子化で使用される演算によるモデルの推論速度の低下が原因であると仮説を立てた。このセクションでは、この主張の真実性を評価するための実験を行う。具体的には、上記にリストされた3つの異なる精度レベルでOpenVLAを再度評価するが、今回はブロッキング制御(blocking control)を使用する。言い換えれば、次の行動が方策によって予測されコントローラーによって実行される前に、各行動がロボット上で完全に実行される。このスキームは、様々な量のレイテンシを持つ手法間でシステムダイナミクスを制御するため、予測速度とは無関係に、方策の行動予測の品質をテストすることができる。事実上、より高いスループットを持つ精度レベルであるbfloat16および4-bit量子化は、8-bit精度でOpenVLAを展開した際に観察されるダイナミクスに一致するように、より遅く実行されることを強制される。したがって、ブロッキング制御下において、8-bit精度のOpenVLAのパフォーマンスは、bfloat16および4-bit精度のパフォーマンスと一致すると予想される。

**実験のセットアップとタスク (Experimental setup and tasks).**
付録D.1および付録D.2で使用されたのと同じ8つのBridgeData V2タスクのサブセットにおいて、ブロッキング制御と量子化された推論を用いたOpenVLAのパフォーマンスを報告する。

**結果 (Results).**
ブロッキング制御を用いた量子化推論実験の結果は表11に示されている。推論速度の低さのために8-bit量子化が最悪のロールアウトパフォーマンスをもたらした表2とは異なり、ここでは、推論速度の変動がタスクパフォーマンスに与える影響を取り除くためにブロッキング制御で評価していることを考慮すると、8-bit量子化はbfloat16精度および4-bit量子化と同等のパフォーマンスを発揮することが観察される。これは、以前の実験（ノンブロッキング制御を使用した場合）における8-bit量子化のパフォーマンスへの推論速度の影響についての我々の仮説を裏付けている。また、セクション5.3でも観察されたように、最も低い精度である4-bitを使用した場合にも大幅なパフォーマンスの低下は見られない。

表11: ブロッキング制御を用いた量子化推論実験結果。推論時にbfloat16精度（デフォルトのアプローチ）、8-bit量子化（int8）、および4-bit量子化（int4）を用いた、様々なBridgeData V2 WidowXタスクでのOpenVLAの成功率と標準誤差を報告する。すべての平均成功率には重なり合うエラーバーがあり、これはすべての手法が同等のパフォーマンスを発揮することを示唆している。

| Category | Task | # Trials | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Visual gen | Put Eggplant into Pot (Easy Version) | 10 | 10 | 10 | 10 |
| Visual gen | Put Eggplant into Pot | 10 | 9 | 10 | 10 |
| Visual gen | Put Cup from Counter into Sink | 10 | 5 | 5 | 3 |
| Motion gen | Lift Eggplant | 10 | 8 | 7 | 7.5 |
| Physical gen | Put Carrot on Plate | 10 | 10 | 10 | 10 |
| Physical gen | Lift AAA Battery | 10 | 3 | 6 | 4 |
| Semantic gen | Take Purple Grapes out of Pot | 10 | 2 | 2 | 2 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot | 10 | 9 | 9.5 | 8.5 |
| | Mean Success Rate | | 70.0 ± 5.1% | 74.4 ± 4.9% | 68.8 ± 5.2% |

---

### E LIBEROシミュレーション実験 (LIBERO Simulation Experiments)
セクション5.2およびセクション5.3における我々の以前の議論は、OpenVLAを新しい実世界のロボットセットアップおよびタスクに適応させることに焦点を当てていた。このセクションでは、OpenVLAをシミュレーションされたロボットのセットアップおよびタスクに適応させることを、具体的にはLIBEROベンチマーク [^116] を利用して探求する。我々のシミュレーションにおける実験は、2つの主な利点を提供する：
1. 汎用性の実証：OpenVLAが、実世界のロボットデータのみで事前学習されているにもかかわらず、実世界とシミュレーション環境およびダイナミクス間の潜在的な差異を克服し、シミュレーションドメインに効果的に適応できることを示す。
2. アクセスしやすさと再現性の向上：公開されているシミュレーションプラットフォームへのOpenVLAの統合は、特にロボットハードウェアへのアクセスを持たない他の研究者にとって、我々のモデルをよりアクセスしやすいものにする。さらに、シミュレーション実験は実世界の実験よりも容易に再現される。

付録E.1で実験のセットアップについて、付録E.2で結果について議論する。実験を再現するために必要な資料はOpenVLAコードベースと共に公開する。

#### E.1 LIBEROシミュレーション実験のセットアップ (LIBERO Simulation Experimental Setup)
**シミュレーションのセットアップとタスク (Simulation setup and tasks).**
LIBEROベンチマーク [^116] は、ロボットマニピュレーションにおける生涯学習を研究するために設計された4つのタスクスイートで構成されており、そのため元の論文では様々なタスクへの前方および後方転移(forward and backward transfer)の両方を調査している。我々の実験では、ターゲットのタスクスイートにおける教師ありファインチューニングのみに焦点を当て、タスクの成功したデモンストレーションに対する行動クローニング(behavioral cloning)によって学習された様々な方策のパフォーマンスを測定する。

我々は以下の4つのタスクスイートで実験を行い、それぞれには50の人間による遠隔操作デモンストレーションを持つ10のタスクが含まれている：
- LIBERO-Spatialは、同じ物体のセットであるが異なるレイアウトで構成され、モデルの空間的関係の理解をテストする。
- LIBERO-Objectは、同じシーンレイアウトであるが異なる物体で構成され、モデルの物体の種類の理解をテストする。
- LIBERO-Goalは、同じ物体とレイアウトであるが異なるタスク目標で構成され、モデルの様々なタスク指向の行動の知識をテストする。
- LIBERO-Long（LIBERO-10とも呼ばれる）は、多様な物体、レイアウト、タスクを持つ長期ホライズン(long-horizon)のタスクで構成される。

我々は上記のそれぞれの学習データセットに以下の変更を加える：
1. より高解像度の画像（例：256×256pxまたは224×224px）を必要とする手法に対応するため、すべてのデモンストレーションを256×256pxの増加した解像度で再生成する。元々、ベンチマークによって提供されるデータセットは128×128pxの画像で構成されている。これらの画像を単に256×256pxにアップスケールするだけでは、画像品質が低下することがわかった。したがって、我々は、必要に応じてダウンスケールできるより高解像度の画像から始めることを選択し、様々な解像度の要件にわたってより高い画像品質を確保する。これらのより高解像度の画像は、人間が収集した提供されたデモンストレーションに保存された行動を用いてシミュレーション環境をステップ実行し、シミュレーターによってレンダリングされた画像を保存することによって得られた。
2. 我々はデータセットからすべての「no-op（何もしない）」行動、すなわち、並進および回転コンポーネントにおける大きさがほぼゼロであり、ロボットのグリッパーの状態を変更しない行動をフィルタリングして除外する。我々は、OpenVLAのような表現力の高いシングルステップ方策にとって、このシンプルなデータクリーニングのステップが重要であることを発見した。そうしなければ、方策はこれらのno-op行動を模倣することを学習し、結果として評価中の特定の状態で無限にフリーズしてしまう。
3. 我々のハードウェア上ではLIBERO環境が上下逆の画像を返すことを観察したため、学習時およびテスト時の両方において、すべての三人称画像を180度回転させる。
4. 我々は模倣学習を介して方策を学習させるため、それはデモンストレーションが成功していることを期待しており、すべてのデモンストレーションを対応するシミュレーション環境でリプレイし、（環境の成功基準によって決定される）タスクの完了に失敗したデモンストレーションをフィルタリングして除外する。結果として、500のLIBERO-Spatialデモンストレーションのうち68件、500のLIBERO-Objectデモンストレーションのうち46件、500のLIBERO-Goalデモンストレーションのうち72件、および500のLIBERO-Longデモンストレーションのうち121件を削除する。
5. 比較するすべての手法について、静的な三人称カメラの画像のみを利用する。元のデータセットで追加で提供されている手首カメラの画像は使用しない。これは公平な比較を行うためであり、OpenVLAの視覚入力は三人称カメラの画像のみで構成されるからである。

**比較 (Comparisons).**
比較する手法には、ゼロから学習されたDiffusion Policy[^8注] [^3]、ターゲットのデータセットでファインチューニングされたOcto [^5]、およびセクション5.3で説明したようにLoRA ( `$r = 32$` ) を介してターゲットのデータセットでファインチューニングされたOpenVLAが含まれる。各方策は（4つのすべてのスイートを組み合わせた単一の方策を学習するのではなく）上記のタスクスイートのそれぞれにおいて独立して学習される。すべての方策は同じデモンストレーションのセットで学習されるため、すべての手法が上記のデータクリーニングのステップの恩恵を受ける。

[^8注]: タスクラベルのDistilBERT [^117] 言語埋め込みによって行動の生成を条件付ける、DROIDデータセットの論文 [^11] で説明されているDiffusion Policyの実装を使用する。

**評価の詳細 (Evaluation details).**
実験結果におけるより低い分散を確保するため、すべての手法は各タスクスイートについて500回の試行で評価され、報告されるパフォーマンスは3つのランダムシード（統計ごとに合計1500回の試行となる）にわたる平均成功率である。前述のように学習データセットを変更するが、テスト環境は変更せず、元のLIBEROベンチマークによって提供されるのと同じ初期環境の構成を使用する。

#### E.2 LIBEROシミュレーション実験結果 (LIBERO Simulation Experimental Results)
我々はLIBEROの実験結果を表12に提示する。重要なことに、OpenVLAはテストされた手法の中で最も高い平均成功率とランクを獲得しており、LIBEROシミュレーション環境のタスクに効果的に適応できることが観察される。しかし、ここではOpenVLAと他の手法との間の全体的な差が、セクション5.2で議論された実世界のファインチューニング実験よりも小さくなっていることがわかった。我々はこれを、OpenVLAがシミュレーションデータなしで純粋に実世界のロボットデータで事前学習されているという事実に起因すると考える。これは、シミュレーション環境と実世界環境およびダイナミクスの間のドメインギャップのため、シミュレーションされたロボットのタスクでモデルをファインチューニングすることは、実世界のタスクでファインチューニングすることほど効果的ではない可能性があることを示唆している。我々は、大量の実世界のロボットデータで事前学習されたもう1つの方策であるOctoによって得られた結果の中に、この考えの証拠を確認している。Octoもまた、ゼロから学習されたDiffusion Policyのようなシンプルで強力なベースラインと比較して、全体的なパフォーマンスにおいてわずかな向上しか達成していない。我々は、もしシミュレーションデータが事前学習のデータ混合に追加されれば、事前学習およびファインチューニングされた手法のパフォーマンスがさらに向上すると予想している。

表12: LIBEROシミュレーションベンチマーク結果。LIBEROベンチマークの4つのタスクスイートについて、それぞれ500回の試行を持つ3つのランダムシードにわたって平均された、各手法の成功率 (SR) と標準誤差を報告する。さらに、各タスクスイート内での各手法のランキングを示す。ランク1はスイート内で最も強力な手法を示し、ランク3は最も弱い手法を示す。（平均ランキングは、様々なタスクのデフォルトとして使用するのに最も適した手法はどれかを知らせてくれるため、注目することが重要である。それは個々のタスクスイートの難易度で正規化されていない平均成功率よりも情報を提供する。）全体として、ファインチューニングされたOpenVLAが最も高い平均成功率とランクを達成し、それにファインチューニングされたOcto、そしてゼロから学習されたDiffusion Policyが続くことがわかった。

| Method | LIBERO-Spatial SR  $(↑)$ | LIBERO-Spatial Rank  $(↓)$ | LIBERO-Object SR  $(↑)$ | LIBERO-Object Rank  $(↓)$ | LIBERO-Goal SR  $(↑)$ | LIBERO-Goal Rank  $(↓)$ | LIBERO-Long SR  $(↑)$ | LIBERO-Long Rank  $(↓)$ | Average SR  $(↑)$ | Average Rank  $(↓)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Diffusion Policy from scratch | 78.3 ± 1.1% | 3 | 92.5 ± 0.7% | 1 | 68.3 ± 1.2% | 3 | 50.5 ± 1.3% | 3 | 72.4 ± 0.7% | 2.5 |
| Octo fine-tuned | 78.9 ± 1.0% | 2 | 85.7 ± 0.9% | 3 | 84.6 ± 0.9% | 1 | 51.1 ± 1.3% | 2 | 75.1 ± 0.6% | 2 |
| OpenVLA fine-tuned (ours) | 84.7 ± 0.9% | 1 | 88.4 ± 0.8% | 2 | 79.2 ± 1.0% | 2 | 53.7 ± 1.3% | 1 | 76.5 ± 0.6% | 1.5 |
