# Colocalisation_microscopy_analysis
Computing spatial statistics for colocalisation quantification from segmented fluorescence microscopy

before using:
input files should be in csv format with the following headers (see segmentation pipeline if starting from raw images xxxaddlinkxxx)
example headers:
label	area	centroid_y	centroid_x	solidity	n_cells_estimate

code can run on excel or csv files

TABLE OF CONTENTS
  1. Overview of Colocalization
  2. Analysis 1: Spatial Overlay
  3. Analysis 2: Nearest-Neighbour Distance
  4. Analysis 3: Grid-Based Intensity Correlation
  5. Analysis 4: Proximity Network
  6. Analysis 5: K-Means Co-Clustering
  7. Analysis 6: DBSCAN Density-Based Clustering
  8. Analysis 7: Ripley's Cross-K / Cross-L Function
  9. Analysis 8: Kernel Density Estimation (KDE) Overlap
  10. Analysis 9: Voronoi Neighbourhood Analysis
  11. Putting It All Together: Interpretation Strategy
  12. References & Further Reading

1.  OVERVIEW OF COLOCALIZATION

  Colocalization analysis asks: do two populations of objects tend to occur
  in the same spatial locations more (or less) than expected by chance?

  There are three possible outcomes:

    POSITIVE COLOCALIZATION (Attraction / Clustering)
      The two populations are found near each other more often than random.
      Example: A receptor and its ligand clustering at the cell membrane.

    RANDOM / INDEPENDENT
      The two populations are distributed independently — knowing where
      one is tells you nothing about where the other is.

    NEGATIVE COLOCALIZATION (Repulsion / Segregation)
      The two populations actively avoid each other — they are found near
      each other less often than random.
      Example: Two transcription factors that mark different cell fates.

  This tool uses 9 complementary methods to assess colocalization. Each
  method captures a different aspect of the spatial relationship. Using
  multiple methods together gives much more confidence than any single test.

  IMPORTANT CONCEPT: COMPLETE SPATIAL RANDOMNESS (CSR)
  Many analyses compare your observed data to "CSR" — what you would
  expect if the objects were scattered completely randomly across the
  region of interest. If your observed values deviate significantly from
  CSR, it suggests a real biological pattern (attraction or repulsion).

2.  ANALYSIS 1: SPATIAL OVERLAY

  FILE: 01_spatial_overlay_*.png

  WHAT IT SHOWS
  Three panels:
    Left:    Only population A (e.g. red dots)
    Centre:  Only population B (e.g. blue dots)
    Right:   Both populations overlaid on the same axes

  Each dot represents one object. The SIZE of each dot is proportional to
  the object's area — larger objects appear as larger dots.

  HOW TO READ IT
  This is your first visual impression. Look for:
    • Do the two populations seem to occupy the same regions?
    • Are there areas where only one population is found?
    • Do they seem to cluster together or stay apart?
    • Is one population more spatially concentrated than the other?

  CALCULATION
  No calculations here — this is purely a visualization. Object areas are
  linearly mapped to marker sizes between 15 and 200 points for display.

  SCALE BAR
  A scale bar (15% of the x-axis range) is shown for spatial reference.
  The units match whatever units your coordinates are in (pixels, µm, etc.).

3.  ANALYSIS 2: NEAREST-NEIGHBOUR DISTANCE

  FILE: 02_nearest_neighbour_*.png

  WHAT IT CALCULATES
  For every object in population A, the tool finds the single closest
  object in population B and measures the distance between them. This is
  called the cross-type nearest-neighbour (NN) distance.

  It does this in both directions:
    A → B:  For each A, find the nearest B
    B → A:  For each B, find the nearest A

  The algorithm uses a KD-tree (a spatial data structure) for efficient
  searching. The distance is Euclidean: d = √((x₁−x₂)² + (y₁−y₂)²).

  METRICS REPORTED
  
  • Mean NN distance (A→B and B→A)
      The average of all nearest-neighbour distances.

  • Median NN distance (A→B and B→A)
      The middle value. Less affected by outliers than the mean.

  • Expected NN distance under CSR
      If population B were scattered randomly at the same density,
      what would you expect the mean NN distance to be?
      Formula: Expected = 0.5 / √(λ)
      where λ = number of B objects / total area of the region.

  • Colocalization Index (observed / expected)
      Ratio of the observed mean NN distance to the CSR expectation.
      
      Interpretation:
        Index < 1  →  Objects are CLOSER than random → ATTRACTION
        Index ≈ 1  →  Objects are at random distances → INDEPENDENT
        Index > 1  →  Objects are FARTHER than random → REPULSION

      Example: Index = 0.6 means objects are on average only 60% as far
      apart as you would expect by chance — strong colocalization.

  FIGURE PANELS
 Left:    Histograms of NN distances for both directions. Dashed lines
           show the CSR expectation. If histograms are shifted left of
           the dashed lines, the populations are closer than random.

  Centre:  Cumulative distribution function (CDF) of NN distances. Shows
           what fraction of objects have their nearest cross-type neighbour
           within a given distance. Steeper curves that rise early mean
           objects are generally very close.

  Right:   Spatial map with coloured lines connecting each A object to its
           nearest B object. Line colour indicates distance (colour bar on
           the right). Short, dark lines in clustered regions = strong
           colocalization.

4.  ANALYSIS 3: GRID-BASED INTENSITY CORRELATION

  FILE: 03_grid_correlation_*.png

  WHAT IT CALCULATES
   The region of interest is divided into a grid (30×30 bins by default).
  For each grid square, the tool counts how many A objects and how many B
  objects fall inside it. This converts your point data into two "density
  images" (like two fluorescence channels).

  Then it computes several correlation and colocalization metrics between
  these two density images:

  PEARSON'S CORRELATION COEFFICIENT (r)
    Formula: r = Σ((Aᵢ - Ā)(Bᵢ - B̄)) / √(Σ(Aᵢ - Ā)² · Σ(Bᵢ - B̄)²)

    where Aᵢ and Bᵢ are the counts in grid square i, and Ā and B̄ are the
    overall means.

    Range: -1 to +1
      r ≈ +1  →  High A where high B (strong colocalization)
      r ≈  0  →  No linear relationship (independent)
      r ≈ -1  →  High A where low B (anti-colocalization)

    The p-value tells you if the correlation is statistically significant
    (p < 0.05 is conventionally considered significant).

  SPEARMAN'S RANK CORRELATION (ρ)
    Like Pearson's, but uses ranks instead of raw values. More robust to
    non-linear relationships and outliers. Interpretation is the same.

  MANDERS' COLOCALIZATION COEFFICIENTS (M1 and M2)
    M1 = fraction of A objects that fall in grid squares where B is present
    M2 = fraction of B objects that fall in grid squares where A is present

    Range: 0 to 1
      M1 = 0   →  None of A overlaps with B
      M1 = 1   →  All of A is found where B is also present
      M1 = 0.7 →  70% of A is in B-positive regions

    Note: M1 and M2 are NOT necessarily equal. You can have a situation
    where most of A is where B is (high M1) but B is also in many places
    where A is not (lower M2).

  COSTES RANDOMIZATION TEST
    Tests whether the observed Pearson r could have arisen by chance.
    The B density image is randomly shuffled 200 times, and the Pearson r
    is recalculated each time. The p-value is the fraction of random r
    values that are ≥ the observed r.

    Costes p < 0.05 → The correlation is unlikely to be due to chance.
    Costes p > 0.05 → The correlation could be a random coincidence.

  FIGURE PANELS
  Panel 1: Density map for population A (red colour scale)
  Panel 2: Density map for population B (blue colour scale)
  Panel 3: Merged RGB image. Red = A only, blue = B only, white/bright =
           overlap. Look for white regions — those are colocalized areas.
  Panel 4: Scatter plot of bin counts. Each dot is one grid square. If
           colocalization is strong, dots will follow a diagonal trend.
           The teal line is the linear regression fit.


5.  ANALYSIS 4: PROXIMITY NETWORK

  FILES: 04_proximity_network_*.png, 04b_degree_distribution_*.png

  WHAT IT CALCULATES
   This analysis builds a NETWORK (graph) where:
    • Nodes = individual objects (A and B)
    • Edges = drawn between an A object and a B object if they are within
              a threshold distance of each other

  The analysis is run at THREE distance thresholds (automatically chosen
  as the 5th, 10th, and 25th percentile of all A-to-B distances):

    Tight threshold   →  Only very close pairs are connected
    Medium threshold  →  Moderately close pairs
    Loose threshold   →  More distant pairs included

  METRICS REPORTED (per threshold)

  • Number of edges
      How many A-B pairs are within the threshold distance.

  • Number of connected components
      How many separate "islands" of connected objects exist. Fewer
      components = more of the network is linked together.

  • Mean degree (A and B separately)
      Average number of cross-type connections per object.
      High degree = that population has many nearby partners.

  • Fraction connected (A and B)
      What percentage of objects have at least one cross-type neighbour.
      If 95% of A objects have at least one B neighbour within the
      threshold, the populations are highly colocalized.

  FIGURE: NETWORK MAP
  Objects are plotted at their spatial positions. Lines connect pairs
  within the threshold. Node colour intensity reflects degree (darker =
  more connections). Three panels show different thresholds.

  FIGURE: DEGREE DISTRIBUTION
  Left:  Histogram showing how many cross-type neighbours each object has.
         If most objects have 0 neighbours → low colocalization.
         If most have many neighbours → high colocalization.
  Right: Degree vs area. Do larger objects tend to have more neighbours?
         This can reveal whether colocalization depends on object size.

6.  ANALYSIS 5: K-MEANS CO-CLUSTERING

  FILES: 05_kmeans_*.png, 05b_kmeans_composition_*.png

  WHAT IT DOES
 
  K-means is an unsupervised clustering algorithm. It groups spatial
  points into k clusters based on their positions, regardless of which
  population they belong to.

  The tool pools ALL objects (A and B together) and runs K-means for
  k = 2, 3, 4, … 10. For each k, it measures:

  ELBOW PLOT
    Shows "inertia" (within-cluster sum of squares) vs k. As k increases,
  inertia always decreases. The "elbow" — where the curve bends — suggests
  a natural number of spatial clusters.

  SILHOUETTE SCORE
  Measures how well-separated the clusters are. Range: -1 to +1.
  Higher = better-defined clusters. The tool picks the k with the
  highest silhouette score as the "best" k.

  MIXING ENTROPY
  THIS IS THE KEY COLOCALIZATION METRIC from K-means.

  For each cluster, the tool counts how many A and how many B objects are
  inside it, then computes Shannon entropy:

    H = -(p_A · log₂(p_A) + p_B · log₂(p_B))

  where p_A = fraction of the cluster that is A, p_B = fraction that is B.

    H = 0   →  The cluster contains ONLY A or ONLY B (segregated)
    H = 1   →  The cluster is 50/50 A and B (perfectly mixed)

  The mixing entropy is averaged across all clusters.

  Interpretation:
    Mean entropy ≈ 1.0  →  Both populations are spatially intermingled
    Mean entropy ≈ 0.0  →  Populations are in separate spatial domains
    Mean entropy ≈ 0.5  →  Moderate mixing

  FIGURE: SPATIAL CLUSTER MAP
  Objects are coloured by their cluster assignment. Circles = A, Squares = B.
  Black X marks = cluster centres. If each cluster contains a mix of
  circles and squares, the populations are colocalized.

  FIGURE: COMPOSITION BAR CHART
  Stacked bars show the A/B composition of each cluster. If all bars are
  roughly half red and half blue, the populations are well-mixed.

7.  ANALYSIS 6: DBSCAN DENSITY-BASED CLUSTERING

  FILE: 06_dbscan_*.png

  WHAT IT DOES
  DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  is a clustering algorithm that finds dense regions of points without
  requiring you to specify the number of clusters.

  Two parameters:
    eps         → neighbourhood radius (automatically estimated from the
                  data as the median 5th-nearest-neighbour distance)
    min_samples → minimum points to form a dense region (default: 5)

  Points in sparse regions are labelled as "noise."

  KEY METRIC: MIXED CLUSTERS
    A cluster is "mixed" if it contains at least one A object AND at least
  one B object. The fraction of mixed clusters tells you how often the
  two populations co-occur in dense regions.

    Fraction mixed ≈ 1.0  →  Every dense region has both A and B
    Fraction mixed ≈ 0.0  →  Dense regions contain only one type
    Fraction mixed ≈ 0.5  →  About half of clusters are co-occupied

  Unlike K-means, DBSCAN:
    • Does not assume spherical clusters
    • Can find clusters of arbitrary shape
    • Identifies noise/outlier points
    • Does not require specifying k

  FIGURE PANELS
  Left:  All clusters coloured differently. Grey X marks = noise points.
  Right: Mixed clusters highlighted in teal with convex hull outlines.
         Single-type or noise points shown in light grey. This immediately
         shows you WHERE colocalization is occurring.

8.  ANALYSIS 7: RIPLEY'S CROSS-K / CROSS-L FUNCTION

  FILE: 07_ripleys_cross_K_*.png

  WHAT IT IS
 
  Ripley's K function is the gold standard spatial statistics method for
  analysing point patterns. The CROSS-K variant (K_ab) specifically measures
  the spatial relationship between two different populations.

  WHAT IT CALCULATES
  For each object in A, the tool counts how many B objects are within
  distance r. This is repeated for many values of r (from 0 to 25% of the
  image extent). The result is normalised by the density of B:

    K_ab(r) = (1 / (n_A · λ_B)) · Σᵢ [number of B within distance r of Aᵢ]

  where λ_B = n_B / area_of_region.

  Under CSR (complete spatial randomness), K(r) = π·r², because in a
  random pattern the expected number of neighbours within distance r
  scales with the area of a circle.

  THE L FUNCTION (variance-stabilised)
  Because K(r) grows as r², it can be hard to interpret visually. The
  L function transforms it:

    L(r) = √(K(r) / π) − r

  Under CSR, L(r) = 0 for all r. Deviations from zero are easy to see:

    L(r) > 0  →  MORE B near A than expected → ATTRACTION at scale r
    L(r) = 0  →  Random distribution at scale r
    L(r) < 0  →  FEWER B near A than expected → REPULSION at scale r

  SIMULATION ENVELOPE
  To test significance, the tool generates 99 random patterns (keeping
  A fixed, randomising B positions) and computes L(r) for each. The 2.5th
  and 97.5th percentiles form a 95% confidence envelope.

  If the observed L(r) (teal line) falls OUTSIDE the grey envelope:
    Above → Statistically significant attraction at that scale
    Below → Statistically significant repulsion at that scale

  METRICS
  • Max deviation: largest amount by which observed L exceeds the mean
    CSR expectation. Larger = stronger colocalization.
  • Mean deviation: average deviation across all scales.

  FIGURE PANELS
  Left:  K_ab(r) with theoretical CSR curve (dashed) and simulation envelope.
  Right: L_ab(r) − r with zero line and simulation envelope. This is the
         most informative panel. Look for the teal line going above the
         grey band — that indicates significant clustering at that spatial
         scale.

  The x-axis (r) tells you AT WHAT SPATIAL SCALE the colocalization occurs.
  For example, if L(r) peaks at r = 20 µm, the two populations tend to
  cluster together at approximately 20 µm scale.


9.  ANALYSIS 8: KERNEL DENSITY ESTIMATION (KDE) OVERLAP
  FILE: 08_kde_overlap_*.png

  WHAT IT DOES
  Instead of treating objects as discrete points, KDE creates a smooth,
  continuous density surface for each population. Think of it as placing
  a small Gaussian "hill" on each object and summing them all up. The
  result is a smooth probability density landscape.

  The tool uses Gaussian KDE with automatic bandwidth selection (Scott's
  rule) as implemented in scipy.

  OVERLAP METRICS
  The two density surfaces (f_A and f_B) are compared:

  Bhattacharyya Overlap Coefficient:
    BC = Σ √(f_A(x,y) · f_B(x,y)) · Δx·Δy

    Range: 0 to 1
      0 → Completely non-overlapping densities
      1 → Identical density distributions
    This is analogous to cos(θ) between two vectors — it measures how
    "aligned" the two density shapes are.

  Min-Overlap (Szymkiewicz-Simpson):
    OV = Σ min(f_A(x,y), f_B(x,y)) · Δx·Δy

    Range: 0 to 1
    This is the area under the lower of the two density curves at each
    point. It represents the fraction of the density that is truly shared.

  FIGURE PANELS
  Left:   KDE contour map for population A (red) with raw points.
  Centre: KDE contour map for population B (blue) with raw points.
  Right:  RGB overlay. Red = A density, blue = B density, white/magenta =
          overlap. Regions that appear white or bright indicate where both
          populations have high density — these are colocalization hotspots.

10.  ANALYSIS 9: VORONOI NEIGHBOURHOOD ANALYSIS

  FILE: 09_voronoi_*.png

  WHAT IT IS
  Voronoi tessellation divides the plane into regions (cells), one per
  object. Each Voronoi cell contains all points closer to that object
  than to any other object. Two objects are "Voronoi neighbours" if their
  cells share an edge.

  This creates a natural, parameter-free definition of "neighbours"
  based on spatial proximity.

  WHAT IT CALCULATES

  For each object, the tool counts what fraction of its Voronoi neighbours
  belong to the OTHER population. This is the "cross-type neighbour
  fraction."

    Cross-type fraction = (# neighbours of other type) / (# total neighbours)

  EXPECTED VALUES UNDER RANDOM LABELLING
  If the labels (A vs B) were assigned randomly to the same set of
  spatial positions, the expected cross-type fraction for A objects would
  be:  (number of B) / (total objects).

  SEGREGATION INDEX
    SI = 1 − (observed cross-type fraction / expected cross-type fraction)

    SI ≈ 0   →  Random mixing (no spatial preference)
    SI > 0   →  Segregation (same-type neighbours preferred)
    SI < 0   →  Attraction (cross-type neighbours preferred = colocalization)

  FIGURE PANELS
  Left:  Voronoi tessellation with edges coloured:
           Teal = edge between A and B (cross-type)
           Grey  = edge between same-type objects
         Nodes coloured by cross-type fraction (green = high, red = low).
         Lots of teal edges = high colocalization.

  Right: Histogram of cross-type fractions for each population, with
         dashed lines showing the random expectation. If the histogram
         is shifted RIGHT of the dashed line, there is more cross-type
         mixing than expected (colocalization). If shifted LEFT,
         populations are segregated.


11.  PUTTING IT ALL TOGETHER: INTERPRETATION STRATEGY

  No single metric perfectly captures colocalization. Here is how to
  combine the results:

  STEP 1: VISUAL INSPECTION
  Start with the spatial overlay (01) and KDE overlap (08). Do the
  populations look like they co-occur? Trust your eyes first, then
  confirm with statistics.

  STEP 2: NEAREST-NEIGHBOUR INDEX (Quick answer)
  The colocalization index (Analysis 2) gives you an immediate
  yes/no/how-much:
    < 1 → Colocalization (attraction)
    ≈ 1 → Random
    > 1 → Segregation (repulsion)

  STEP 3: RIPLEY'S L FUNCTION (Rigorous statistical test)
  This is your most rigorous test. If the L(r) curve goes above the
  95% envelope, you have statistically significant colocalization.
  The r value where the peak occurs tells you the spatial scale.

  STEP 4: GRID CORRELATION (Traditional colocalization)
  Pearson r and Manders M1/M2 are widely reported in biology papers.
  Include these for comparability with other studies. Check the Costes
  p-value to confirm significance.

  STEP 5: CLUSTERING ANALYSES (Spatial structure)
  K-means mixing entropy and DBSCAN mixed clusters tell you about
  the spatial structure of colocalization:
    • Are the populations intermingled everywhere? (High mixing)
    • Or colocalized only in specific hotspots? (Mixed clusters)

  STEP 6: NETWORK & VORONOI (Object-level detail)
  The proximity network and Voronoi analysis give object-level detail:
    • Which specific objects are colocalized?
    • How many cross-type partners does each object have?
    • Is there a relationship between object size and colocalization?

  
12.  REFERENCES & FURTHER READING

  Nearest-Neighbour Analysis:
    Clark, P.J. & Evans, F.C. (1954). Distance to nearest neighbor as a
    measure of spatial relationships in populations. Ecology, 35(4), 445–453.

  Pearson Correlation / Manders Coefficients:
    Manders, E.M.M., Verbeek, F.J. & Aten, J.A. (1993). Measurement of
    co-localization of objects in dual-colour confocal images. Journal of
    Microscopy, 169(3), 375–382.

  Costes Randomization:
    Costes, S.V. et al. (2004). Automatic and quantitative measurement of
    protein-protein colocalization in live cells. Biophysical Journal,
    86(6), 3993–4003.

  Ripley's K Function:
    Ripley, B.D. (1976). The second-order analysis of stationary point
    processes. Journal of Applied Probability, 13(2), 255–266.

    Diggle, P.J. (2003). Statistical Analysis of Spatial Point Patterns.
    2nd ed., Arnold.

  DBSCAN:
    Ester, M. et al. (1996). A density-based algorithm for discovering
    clusters in large spatial databases with noise. KDD-96 Proceedings.

  K-Means:
    Lloyd, S.P. (1982). Least squares quantization in PCM. IEEE
    Transactions on Information Theory, 28(2), 129–137.

  Voronoi Tessellation:
    Okabe, A. et al. (2000). Spatial Tessellations: Concepts and
    Applications of Voronoi Diagrams. 2nd ed., Wiley.

  KDE:
    Silverman, B.W. (1986). Density Estimation for Statistics and Data
    Analysis. Chapman & Hall.

  General Colocalization Review:
    Dunn, K.W., Kamocka, M.M. & McDonald, J.H. (2011). A practical
    guide to evaluating colocalization in biological microscopy. American
    Journal of Physiology — Cell Physiology, 300(4), C723–C742.

  Bhattacharyya Coefficient:
    Bhattacharyya, A. (1943). On a measure of divergence between two
    statistical populations defined by their probability distributions.
    Bulletin of the Calcutta Mathematical Society, 35, 99–109.
