package elki.index.idistance;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import elki.clustering.kmedoids.initialization.KMedoidsInitialization;
import elki.data.type.TypeInformation;
import elki.database.ids.ArrayDBIDs;
import elki.database.ids.DBID;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DoubleDBIDListIter;
import elki.database.ids.KNNHeap;
import elki.database.ids.KNNList;
import elki.database.ids.ModifiableDoubleDBIDList;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.knn.KNNSearcher;
import elki.database.query.range.RangeSearcher;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.AbstractRefiningIndex;
import elki.index.IndexFactory;
import elki.index.KNNIndex;
import elki.index.RangeIndex;
import elki.logging.Logging;
import elki.logging.statistics.DoubleStatistic;
import elki.logging.statistics.LongStatistic;
import elki.math.MeanVarianceMinMax;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.pairs.DoubleIntPair;

public class InMemoryMIndex<O> extends AbstractRefiningIndex<O> implements RangeIndex<O>, KNNIndex<O> {

  /**
   * Class logger.
   */
  private static final Logging LOG = Logging.getLogger(InMemoryMIndex.class);

  /**
   * Distance query.
   */
  protected DistanceQuery<O> distanceQuery;

  /**
   * Initialization method.
   */
  protected KMedoidsInitialization<O> initialization;

  /**
   * Number of reference points.
   */
  protected int numberOfReferencePoints;

  protected int numberOfStoredDistances;

  /**
   * Reference points.
   */
  protected ArrayDBIDs referencePoints;

  /**
   * The index.
   */
  protected ModifiableDoubleDBIDList[] mIndex;

  /**
   * Map that stores a distanceIndex for each DBID object.
   */
  protected Map<DBID, Integer> indexMap;

  /**
   * Distances from all DBID objects to their closest reference points.
   */
  protected double[][] distancesToReferencePoints;

  /**
   * Index of the reference point corresponding to a distance in
   * distanceToReferencepoints
   */
  protected int[][] referencePointIDs;

  protected double rmin[];

  protected double rmax[];

  public InMemoryMIndex(Relation<O> relation, DistanceQuery<O> distance, KMedoidsInitialization<O> initialization, int numberOfReferencePoints, int numberOfStoredDistances) {
    super(relation);
    this.distanceQuery = distance;
    this.initialization = initialization;
    this.numberOfReferencePoints = numberOfReferencePoints;
    if(!distance.getDistance().isMetric()) {
      LOG.warning("The M-Index assumes a metric distance functions.\n" //
          + distance.getDistance().getClass() + " does not report itself as metric.\n" //
          + "M-Index will run, but may yield approximate results.");
    }
    assert (numberOfStoredDistances > 0 && numberOfStoredDistances <= numberOfReferencePoints);
    this.numberOfStoredDistances = numberOfStoredDistances;

  }

  @Override
  public void initialize() {
    // initialize reference points
    referencePoints = DBIDUtil.ensureArray(initialization.chooseInitialMedoids(numberOfReferencePoints, relation.getDBIDs(), distanceQuery));

    final int k = referencePoints.size();
    final int n = relation.size();
    int counter = 0;

    mIndex = new ModifiableDoubleDBIDList[k];
    distancesToReferencePoints = new double[n][numberOfStoredDistances];
    referencePointIDs = new int[n][numberOfStoredDistances];

    rmin = new double[numberOfReferencePoints];
    rmax = new double[numberOfReferencePoints];
    indexMap = new HashMap<>();
    assert (k == numberOfReferencePoints);

    // Create a list for each cluster
    for(int i = 0; i < k; i++) {
      mIndex[i] = DBIDUtil.newDistanceDBIDList(relation.size() / (2 * k));
    }

    DBIDArrayIter referencePointIterator = referencePoints.iter();

    // Compute distances for each data point
    for(DBIDIter dataPointIterator = relation.iterDBIDs(); dataPointIterator.valid(); dataPointIterator.advance()) {

      DoubleIntPair[] currentDistances = new DoubleIntPair[k];

      // Compute distances from current data point to all reference points
      for(referencePointIterator.seek(0); referencePointIterator.valid(); referencePointIterator.advance()) {

        final int i = referencePointIterator.getOffset();
        double distance = distanceQuery.distance(referencePointIterator, dataPointIterator);
        currentDistances[i] = new DoubleIntPair(distance, i);
      }
      Arrays.sort(currentDistances);
      mIndex[currentDistances[0].second].add(currentDistances[0].first, dataPointIterator);
      for(int j = 0; j < numberOfStoredDistances; j++) {
        distancesToReferencePoints[counter][j] = currentDistances[j].first;
        referencePointIDs[counter][j] = currentDistances[j].second;
      }
      // Add distanceIndex of current data point to map
      indexMap.put(DBIDUtil.deref(dataPointIterator), counter);
      counter++;
    }

    for(int i = 0; i < k; i++) {
      mIndex[i].sort();
      rmin[i] = mIndex[i].doubleValue(0);
      rmax[i] = mIndex[i].doubleValue(mIndex[i].size() - 1);
    }

  }

  @Override
  public KNNSearcher<O> kNNByObject(DistanceQuery<O> distanceQuery, int maxk, int flags) {
    return distanceQuery.getRelation() == relation && this.getDistance().equals(distanceQuery.getDistance()) ? //
        new MIndexKNNSearcher(distanceQuery) : null;
  }

  @Override
  public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxradius, int flags) {
    return distanceQuery.getRelation() == relation && this.getDistance().equals(distanceQuery.getDistance()) ? //
        new MIndexRangeSearcher(distanceQuery) : null;
  }

  private Distance<? super O> getDistance() {
    return distanceQuery.getDistance();
  }

  @Override
  public Logging getLogger() {
    return LOG;
  }

  @Override
  public void logStatistics() {
    super.logStatistics();
    MeanVarianceMinMax mm = new MeanVarianceMinMax();
    for(int i = 0; i < mIndex.length; i++) {
      mm.put(mIndex[i].size());
    }
    LOG.statistics(new LongStatistic(InMemoryMIndex.class.getName() + ".size.min", (int) mm.getMin()));
    LOG.statistics(new DoubleStatistic(InMemoryMIndex.class.getName() + ".size.mean", mm.getMean()));
    LOG.statistics(new LongStatistic(InMemoryMIndex.class.getName() + ".size.max", (int) mm.getMax()));
  }

  /**
   * Computes and sorts the distances from query object to reference points.
   * 
   * @param distanceQuery   Distance function
   * @param queryObject     Query object
   * @param referencePoints Reference points
   */
  protected static <O> DoubleIntPair[] computeDistanceToReferencePoints(DistanceQuery<O> distanceQuery, O queryObject, ArrayDBIDs referencePoints) {
    DoubleIntPair[] referencePointDistances = new DoubleIntPair[referencePoints.size()];
    // Compute distances to reference points.
    for(DBIDArrayIter referencePointIterator = referencePoints.iter(); referencePointIterator.valid(); referencePointIterator.advance()) {
      final int i = referencePointIterator.getOffset();
      final double dist = distanceQuery.distance(queryObject, referencePointIterator);
      referencePointDistances[i] = new DoubleIntPair(dist, i);
    }
    return referencePointDistances;
  }

  /**
   * Seek an iterator to the first element with a value >= radius
   * 
   * @param index              Index to search
   * @param partitionInterator Iterator for a partition
   * @param radius             Distance to search to
   */
  protected static void binarySearch(ModifiableDoubleDBIDList index, DoubleDBIDListIter partitionIterator, double radius) {
    int lowerBound = 0, upperBound = index.size() - 1;
    while(lowerBound < upperBound) {
      final int medianPosition = (lowerBound + upperBound) >>> 1;
      final double medianValue = partitionIterator.seek(medianPosition).doubleValue();

      if(medianValue >= radius) {
        upperBound = medianPosition;
      }
      else {
        lowerBound = medianPosition + 1;
      }
    }
    partitionIterator.seek(lowerBound);
    partitionIterator.advance();

  }

  /**
   * Compute max |d(o,pi) - d(q,pi)| over all object pivot distances stored
   * 
   * @param objectIndex               Index of the object
   * @param distanceToReferencePoints Object-reference point distances
   * @param referencePointIDs         IDs of the reference points, that object
   *                                  reference point distances are stored for
   * @param referencePointDistances   Distances from queryObject to all
   *                                  reference points
   */
  protected static double pivotFilteringMaxDeviation(int objectIndex, double[][] distancesToReferencePoints, int[][] referencePointIDs, DoubleIntPair[] referencePointDistances) {
    double maxValue = 0;
    final int numberOfStoredDistances = referencePointIDs[0].length;
    for(int j = 0; j < numberOfStoredDistances; j++) {
      final int currentReferencePoint = referencePointIDs[objectIndex][j];
      final double currentObjectReferencePointDistance = distancesToReferencePoints[objectIndex][j];
      double currentValue = Math.abs(currentObjectReferencePointDistance - referencePointDistances[currentReferencePoint].first);
      if(currentValue > maxValue)
        maxValue = currentValue;
    }
    return maxValue;
  }

  /**
   * kNN query implementation.
   * 
   * @author Lennart Nöller
   */
  protected class MIndexKNNSearcher extends AbstractRefiningIndex<O>.AbstractRefiningQuery implements KNNSearcher<O> {
    /**
     * Constructor.
     * 
     * @param distanceQuery Distance query
     */
    public MIndexKNNSearcher(DistanceQuery<O> distanceQuery) {
      super(distanceQuery);
    }

    @Override
    public KNNList getKNN(O queryObject, int k) {
      // Array used for Pivot Filtering
      DoubleIntPair[] referencePointDistances = computeDistanceToReferencePoints(distanceQuery, queryObject, referencePoints);
      // Array used for iterating over clusters
      DoubleIntPair[] rankedReferencePoints = Arrays.copyOf(referencePointDistances, numberOfReferencePoints);
      Arrays.sort(rankedReferencePoints);

      KNNHeap kNNHeap = DBIDUtil.newHeap(k);
      double kDistanceUpperBound = Double.POSITIVE_INFINITY;
      for(DoubleIntPair currentPair : rankedReferencePoints) {
        final int pivot = currentPair.second;
        final double queryPivotDistance = currentPair.first;
        
        // Double Pivot Distance Constraint
        if((queryPivotDistance - rankedReferencePoints[0].first) > 2 * kDistanceUpperBound)
          break;

        double lowerDistanceBound = queryPivotDistance - kDistanceUpperBound;
        double upperDistanceBound = queryPivotDistance + kDistanceUpperBound;

        // Range Pivot Distance Constraint
        // Clusters that have no objects which could be in the search sphere
        if(upperDistanceBound < rmin[pivot])
          continue;
        if(lowerDistanceBound > rmax[pivot])
          continue;

        final ModifiableDoubleDBIDList currentPartition = mIndex[pivot];
        final DoubleDBIDListIter forwardIterator = currentPartition.iter(),
            backwardIterator = currentPartition.iter();

        // move iterator to first object with a larger distance to its
        // pivot than kDistanceUpperBound
        binarySearch(currentPartition, forwardIterator, queryPivotDistance);
        backwardIterator.seek(forwardIterator.getOffset() - 1);

        boolean searchForward = forwardIterator.valid();
        boolean searchBackward = backwardIterator.valid();

        DBID nextObject;
        // consider all objects within [d(q,pi) - kDistanceUpperBound, d(q,pi) + kDistanceUpperBound]
        while(searchForward || searchBackward) {
          // keep searching towards pivot
          if(!searchForward) {
            nextObject = DBIDUtil.deref(backwardIterator);
            backwardIterator.retract();
          }
          // keep searching away from pivot
          else if(!searchBackward) {
            nextObject = DBIDUtil.deref(forwardIterator);
            forwardIterator.advance();
          }
          // consider object with smaller deviation of their distance to their
          // pivot from kDistanceUpperBound
          else {
            final double deviationForward = Math.abs(forwardIterator.doubleValue() - queryPivotDistance);
            final double deviationBackward = Math.abs(backwardIterator.doubleValue() - queryPivotDistance);

            if(deviationForward < deviationBackward) {
              nextObject = DBIDUtil.deref(forwardIterator);
              forwardIterator.advance();
            }
            else {
              nextObject = DBIDUtil.deref(backwardIterator);
              backwardIterator.retract();
            }
          }
          // pivot filtering 
          final int objectIndex = indexMap.get(nextObject);
          double maxValue = pivotFilteringMaxDeviation(objectIndex, distancesToReferencePoints, referencePointIDs, referencePointDistances);
          if(maxValue <= kDistanceUpperBound) {
            final double distance = refine(nextObject, queryObject);
            if(distance < kDistanceUpperBound) {
              kNNHeap.insert(distance, nextObject);
              kDistanceUpperBound = kNNHeap.getKNNDistance();
              
              //adjust bounds
              lowerDistanceBound = queryPivotDistance - kDistanceUpperBound;
              upperDistanceBound = queryPivotDistance + kDistanceUpperBound;
            }
          }
          // check if forward search continues
          if(searchForward) {
            if(!forwardIterator.valid())
              searchForward = false;
            else if(forwardIterator.doubleValue() > upperDistanceBound)
              searchForward = false;
          }
          // check if backward search continues
          if(searchBackward) {
            if(!backwardIterator.valid())
              searchBackward = false;
            else if(backwardIterator.doubleValue() < lowerDistanceBound)
              searchBackward = false;
          }
        }
      }
      KNNList list = kNNHeap.toKNNList();
      return list;

    }
  }

  /**
   * Exact range query implementation.
   * 
   * @author Lennart Nöller
   */
  protected class MIndexRangeSearcher extends AbstractRefiningIndex<O>.AbstractRefiningQuery implements RangeSearcher<O> {
    /**
     * Constructor.
     * 
     * @param distanceQuery Distance query
     */
    public MIndexRangeSearcher(DistanceQuery<O> distanceQuery) {
      super(distanceQuery);
    }

    @Override
    public ModifiableDoubleDBIDList getRange(O queryObject, double searchRadius, ModifiableDoubleDBIDList result) {
      // Array used for Pivot Filtering
      DoubleIntPair[] referencePointDistances = computeDistanceToReferencePoints(distanceQuery, queryObject, referencePoints);
      // Array used for iterating over clusters
      DoubleIntPair[] rankedReferencePoints = Arrays.copyOf(referencePointDistances, numberOfReferencePoints);
      Arrays.sort(rankedReferencePoints);

      DBIDUtil.ensureArray(relation.getDBIDs());

      for(DoubleIntPair currentPair : rankedReferencePoints) {
        final int pivot = currentPair.second;
        final double queryPivotDistance = currentPair.first;

        // Double Pivot Distance Constraint
        // Since reference points are ordered we can stop when the condition is
        // first fulfilled
        if((queryPivotDistance - rankedReferencePoints[0].first) > 2 * searchRadius)
          break;

        final double lowerDistanceBound = queryPivotDistance - searchRadius;
        final double upperDistanceBound = queryPivotDistance + searchRadius;

        // Range Pivot Distance Constraint
        if(upperDistanceBound < rmin[pivot])
          continue;
        if(lowerDistanceBound > rmax[pivot])
          continue;

        final ModifiableDoubleDBIDList currentPartition = mIndex[pivot];
        final DoubleDBIDListIter dataPointIterator = currentPartition.iter();

        // move iterator to first object within distance range
        binarySearch(currentPartition, dataPointIterator, lowerDistanceBound);
        if(!dataPointIterator.valid() || dataPointIterator.doubleValue() > upperDistanceBound)
          continue;
        double currentObjectDistance = dataPointIterator.doubleValue();

        // consider all objects within [d(q,pi) - r, d(q,pi) + r] range
        while(dataPointIterator.valid()) {
          // stop searching cluster if distances exceed upper bound
          if(currentObjectDistance > upperDistanceBound)
            break;

          // Pivot Filtering
          final int objectIndex = indexMap.get(DBIDUtil.deref(dataPointIterator));
          double maxValue = pivotFilteringMaxDeviation(objectIndex, distancesToReferencePoints, referencePointIDs, referencePointDistances);
          if(maxValue > searchRadius) {
            dataPointIterator.advance();
            continue;
          }

          // Compute actual distance
          final double distance = refine(dataPointIterator, queryObject);
          if(distance <= searchRadius)
            result.add(distance, dataPointIterator);
          dataPointIterator.advance();
          currentObjectDistance = dataPointIterator.doubleValue();
        }
      }
      return result;
    }
  }

  /**
   * Index factory for M-Index indexes.
   * 
   * @author Lennart Nöller
   * 
   * @has - - - InMemoryMIndex
   * 
   * @param <V> Data type.
   */
  public static class Factory<V> implements IndexFactory<V> {
    /**
     * Distance function to use.
     */
    Distance<? super V> distance;

    /**
     * Initialization method.
     */
    KMedoidsInitialization<V> initialization;

    /**
     * Number of reference points
     */
    int k;

    /**
     * Number of distances to be stored per data point
     */
    int numberOfStoredDistances;

    /**
     * Constructor.
     * 
     * @param distance                Distance function
     * @param initialization          Initialization method
     * @param k                       Number of reference points
     * @param numberOFStoredDistances Number of distances to be stored per data
     *                                point
     */
    public Factory(Distance<? super V> distance, KMedoidsInitialization<V> initialization, int k, int numberOfStoredDistances) {
      super();
      this.distance = distance;
      this.initialization = initialization;
      this.k = k;
      this.numberOfStoredDistances = numberOfStoredDistances;
    }

    @Override
    public InMemoryMIndex<V> instantiate(Relation<V> relation) {
      return new InMemoryMIndex<>(relation, distance.instantiate(relation), initialization, k, numberOfStoredDistances);
    }

    @Override
    public TypeInformation getInputTypeRestriction() {
      return distance.getInputTypeRestriction();
    }

    /**
     * Parameterization class.
     * 
     * @author Lennart Nöller
     * 
     * @hidden
     * 
     * @param <V> object type.
     */
    public static class Par<V> implements Parameterizer {
      /**
       * Parameter for the distance function
       */
      public static final OptionID DISTANCE_ID = new OptionID("M-Index.distance", "Distance function to build the index for.");

      /**
       * Initialization method.
       */
      public static final OptionID REFERENCE_ID = new OptionID("M-Index.reference", "Method to choose the reference points.");

      /**
       * Number of reference points.
       */
      public static final OptionID K_ID = new OptionID("M-Index.k", "Number of reference points to use.");

      /**
       * Number of distances stored per data point.
       */
      public static final OptionID STORED_DISTANCES_ID = new OptionID("M-Index.stored_distances", "Number of distances stored per data point.");

      /**
       * Distance function to use.
       */
      Distance<? super V> distance;

      /**
       * Initialization method.
       */
      KMedoidsInitialization<V> initialization;

      /**
       * Number of reference points
       */
      int k;

      int numberOfStoredDistances;

      @Override
      public void configure(Parameterization config) {
        new ObjectParameter<Distance<? super V>>(DISTANCE_ID, Distance.class) //
            .grab(config, x -> distance = x);
        new ObjectParameter<KMedoidsInitialization<V>>(REFERENCE_ID, KMedoidsInitialization.class) //
            .grab(config, x -> initialization = x);
        new IntParameter(K_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> k = x);
        new IntParameter(STORED_DISTANCES_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> numberOfStoredDistances = x);
      }

      @Override
      public InMemoryMIndex.Factory<V> make() {
        return new InMemoryMIndex.Factory<>(distance, initialization, k, numberOfStoredDistances);
      }
    }
  }
}
