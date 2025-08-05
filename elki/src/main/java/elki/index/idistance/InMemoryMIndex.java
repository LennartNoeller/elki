package elki.index.idistance;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import elki.clustering.kmedoids.initialization.KMedoidsInitialization;
import elki.data.type.TypeInformation;
import elki.database.ids.ArrayDBIDs;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DoubleDBIDListIter;
import elki.database.ids.KNNHeap;
import elki.database.ids.KNNList;
import elki.database.ids.ModifiableDoubleDBIDList;
import elki.database.ids.DBID;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.knn.KNNSearcher;
import elki.database.query.range.RangeSearcher;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.AbstractRefiningIndex;
import elki.index.IndexFactory;
import elki.index.KNNIndex;
import elki.index.RangeIndex;
import elki.index.idistance.InMemoryIDistanceIndex.IDistanceKNNSearcher;
import elki.index.idistance.InMemoryIDistanceIndex.IDistanceRangeSearcher;
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

public class InMemoryMIndex<O> extends AbstractRefiningIndex<O> implements RangeIndex<O>, KNNIndex<O>{
  
  /**
   * Class logger.
   */
  private static final Logging LOG = Logging.getLogger(InMemoryMIndex.class);

  /**
   * Distance query.
   */
  private DistanceQuery<O> distanceQuery;

  /**
   * Initialization method.
   */
  private KMedoidsInitialization<O> initialization;

  /**
   * Number of reference points.
   */
  private int numberOfReferencePoints;
  
  private int numberOfStoredDistances;

  /**
   * Reference points.
   */
  private ArrayDBIDs referencePoints;

  /**
   * The index.
   */
  private ModifiableDoubleDBIDList[] mIndex;
  
  /**
   * Map that stores a distanceIndex for each DBID object.
   */
  private Map<DBIDRef, Integer> indexMap;
  
  /**
   * Distances from all DBID objects to their closest reference points. 
   */
  private double[][] distancesToReferencePoints;
  
  /**
   * Index of the reference point corresponding to a distance in distanceToReferencepoints
   */
  private int[][] referencePointIDs;
  
  
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
    assert(numberOfStoredDistances > 0 && numberOfStoredDistances <= numberOfReferencePoints);
    this.numberOfStoredDistances = numberOfStoredDistances;
   
  }
  
  @Override
  public void initialize(){
    //initialize reference points
    referencePoints = DBIDUtil.ensureArray(initialization.chooseInitialMedoids(numberOfReferencePoints, relation.getDBIDs(), distanceQuery));
    
    final int k = referencePoints.size();
    final int n = relation.size();
    int counter = 0;
    
    mIndex = new ModifiableDoubleDBIDList[k];
    distancesToReferencePoints = new double[n][numberOfStoredDistances];
    referencePointIDs = new int[n][numberOfStoredDistances];    
    indexMap = new HashMap<>();
    assert(k == numberOfReferencePoints);
    
    //Create a list for each cluster
    for(int i = 0; i < k; i++ ) {
      mIndex[i] = DBIDUtil.newDistanceDBIDList(relation.size() / (2 * k));
    }
    
    DBIDArrayIter referencePointIterator = referencePoints.iter();
    
    //Compute distances for each data point
    for(DBIDIter dataPointIterator = relation.iterDBIDs(); dataPointIterator.valid(); dataPointIterator.advance()) {
      
      DoubleIntPair[] currentDistances = new DoubleIntPair[k];
      
      //Compute distances from current data point to all reference points
      for(referencePointIterator.seek(0);referencePointIterator.valid();referencePointIterator.advance()) {
        
        final int i = referencePointIterator.getOffset();
        double distance = distanceQuery.distance(referencePointIterator, dataPointIterator);
        currentDistances[i] = new DoubleIntPair(distance, i);
      }
      Arrays.sort(currentDistances);
      mIndex[currentDistances[0].second].add(currentDistances[0].first,dataPointIterator);
      for(int j = 0; j < numberOfStoredDistances; j++) {
        distancesToReferencePoints[counter][j] = currentDistances[j].first;
        referencePointIDs[counter][j] = currentDistances[j].second;
      }
      //Add distanceIndex of current data point to map
      indexMap.put(dataPointIterator, counter);
      counter++;
    }
    
    for(int i = 0; i < k; i++) {
      mIndex[i].sort();
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
  
  protected static <O> DoubleIntPair[] rankReferencePoints(DistanceQuery<O> distanceQuery, O obj, ArrayDBIDs referencepoints) {
    DoubleIntPair[] priority = new DoubleIntPair[referencepoints.size()];
    // Compute distances to reference points.
    for(DBIDArrayIter iter = referencepoints.iter(); iter.valid(); iter.advance()) {
      final int i = iter.getOffset();
      final double dist = distanceQuery.distance(obj, iter);
      priority[i] = new DoubleIntPair(dist, i);
    }
    Arrays.sort(priority);
    return priority;
  }

  /**
   * Seek an iterator to the desired position, using binary search.
   * 
   * @param index Index to search
   * @param iter Iterator
   * @param val Distance to search to
   */
  protected static void binarySearch(ModifiableDoubleDBIDList index, DoubleDBIDListIter iter, double val) {
    // Binary search. TODO: move this into the DoubleDBIDList class.
    int left = 0, right = index.size();
    while(left < right) {
      final int mid = (left + right) >>> 1;
      final double curd = iter.seek(mid).doubleValue();
      if(val < curd) {
        right = mid;
      }
      else if(val > curd) {
        left = mid + 1;
      }
      else {
        left = mid;
        break;
      }
    }
    if(left >= index.size()) {
      --left;
    }
    iter.seek(left);
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
    public KNNList getKNN(O obj, int k) {
      
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
    public ModifiableDoubleDBIDList getRange(O obj, double range, ModifiableDoubleDBIDList result) {
     
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
     * @param distance Distance function
     * @param initialization Initialization method
     * @param k Number of reference points
     * @param numberOFStoredDistances Number of distances to be stored per data point
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
