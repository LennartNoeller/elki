package elki.index.idistance;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Comparator;

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

public class InMemoryDynamicMIndex<O> extends AbstractRefiningIndex<O> implements RangeIndex<O>, KNNIndex<O> {

  /**
   * Class logger.
   */
  private static final Logging LOG = Logging.getLogger(InMemoryDynamicMIndex.class);

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
   * The index cluster tree.
   */
  private TreeNode[] dynamicClusterTree;

  /**
   * Map that stores a distanceIndex for each DBID object.
   */
  private Map<DBID, Integer> indexMap;

  /**
   * Distances from all DBID objects to their closest reference points.
   */
  private double[][] distancesToReferencePoints;

  /**
   * Index of the reference point corresponding to a distance in
   * distanceToReferencepoints
   */
  private int[][] referencePointIDs;

  /**
   * Maximum number of layers
   */
  private int lmax;

  /**
   * Maximum number of objects in non lmax clusters
   */
  private int maxClusterSize;

  public InMemoryDynamicMIndex(Relation<O> relation, DistanceQuery<O> distance, KMedoidsInitialization<O> initialization, int numberOfReferencePoints, int lmax, int maxClusterSize, int numberOfStoredDistances) {
    super(relation);
    this.distanceQuery = distance;
    this.initialization = initialization;
    this.numberOfReferencePoints = numberOfReferencePoints;
    if(!distance.getDistance().isMetric()) {
      LOG.warning("The M-Index assumes a metric distance functions.\n" //
          + distance.getDistance().getClass() + " does not report itself as metric.\n" //
          + "M-Index will run, but may yield approximate results.");
    }
    assert (lmax > 0 && lmax <= numberOfReferencePoints);
    this.lmax = lmax;
    assert (maxClusterSize >= 0);
    this.maxClusterSize = maxClusterSize;
    assert (numberOfStoredDistances > 1 && numberOfStoredDistances <= numberOfReferencePoints);
    this.numberOfStoredDistances = numberOfStoredDistances;

  }

  @Override
  public void initialize() {
    // initialize reference points
    referencePoints = DBIDUtil.ensureArray(initialization.chooseInitialMedoids(numberOfReferencePoints, relation.getDBIDs(), distanceQuery));

    final int k = referencePoints.size();
    final int n = relation.size();
    int counter = 0;

    dynamicClusterTree = new TreeNode[k];
    ModifiableDoubleDBIDList[] mIndex = new ModifiableDoubleDBIDList[k];
    distancesToReferencePoints = new double[n][numberOfStoredDistances];
    referencePointIDs = new int[n][numberOfStoredDistances];

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

    // create cluster tree
    for(int i = 0; i < k; i++) {
      LeafNode newNode = new LeafNode(i, i, 0, mIndex[i].doubleValue(0), mIndex[i].doubleValue(mIndex[i].size() - 1), mIndex[i]);
      if(lmax > 1 && newNode.size > maxClusterSize) {
        dynamicClusterTree[i] = recursiveSplit(newNode);
        continue;
      }
      dynamicClusterTree[i] = newNode;
    }
  }

  @SuppressWarnings("unchecked")
  private InternalNode recursiveSplit(LeafNode node) {
    InternalNode newNode = node.splitCluster();
    if(newNode.level < lmax - 1) {
      for(int i = 0; i < newNode.children.length; i++) {
        TreeNode child = newNode.children[i];
        if(child.size > maxClusterSize) {
          if(child instanceof InMemoryDynamicMIndex.LeafNode)
            newNode.children[i] = recursiveSplit((LeafNode) child);
        }
      }
    }
    return newNode;
  }

  @Override
  public KNNSearcher<O> kNNByObject(DistanceQuery<O> distanceQuery, int maxk, int flags) {
    return distanceQuery.getRelation() == relation && this.getDistance().equals(distanceQuery.getDistance()) ? //
        new DynamicMIndexKNNSearcher(distanceQuery) : null;
  }

  @Override
  public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxradius, int flags) {
    return distanceQuery.getRelation() == relation && this.getDistance().equals(distanceQuery.getDistance()) ? //
        new DynamicMIndexRangeSearcher(distanceQuery) : null;
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
    for(int i = 0; i < dynamicClusterTree.length; i++) {
      mm.put(dynamicClusterTree[i].size);
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
   * For each reference point check if d(q,pi) - d(q,p0) <= 2 * range holds
   * 
   * @param referencePointDisntaces distance from query object to all reference
   *                                points
   * @param closestReferencePoint   ID of the reference point closest to q
   * @param range                   search radius
   */
  protected static boolean[] doublePivotDistanceConstraint(DoubleIntPair[] referencePointDistances, int closestReferencePoint, double range) {
    int k = referencePointDistances.length;
    boolean[] inDoublePivotDistanceConstraint = new boolean[k];
    for(int i = 0; i < k; i++) {
      if(referencePointDistances[i].first - referencePointDistances[closestReferencePoint].first <= 2 * range)
        inDoublePivotDistanceConstraint[i] = true;
    }
    return inDoublePivotDistanceConstraint;
  }

  private static abstract class TreeNode {
    int currentReferencePointIndex;

    int size;

    int level;

    double rmin;

    double rmax;

    public TreeNode(int currentReferencePoint, int size, int level, double rmin, double rmax) {
      this.currentReferencePointIndex = currentReferencePoint;
      this.size = size;
      this.level = level;
      this.rmin = rmin;
      this.rmax = rmax;
    }
  }

  private class InternalNode extends TreeNode {
    TreeNode[] children;

    public InternalNode(int currentReferencePoint, int size, int level, double rmin, double rmax, TreeNode[] children) {
      super(currentReferencePoint, size, level, rmin, rmax);
      this.children = children;
    }
  }

  private class LeafNode extends TreeNode {
    int primaryReferencePointIndex;
    ModifiableDoubleDBIDList objects;

    public LeafNode(int currentReferencePoint,int primaryReferencePoint, int level, double rmin, double rmax, ModifiableDoubleDBIDList objects) {

      super(currentReferencePoint, objects.size(), level, rmin, rmax);
      this.primaryReferencePointIndex = primaryReferencePoint;
      this.objects = objects;
    }

    // returns a new internal node with new leaf nodes
    public InternalNode splitCluster() {
      int numberOfNewClusters = 0;
      // number of objects with next closest reference point for each reference
      // point
      final int[] newClusterSizes = new int[numberOfReferencePoints];
      // object id for each object in current node
      final int[] objectIDs = new int[size];
      // index of each reference points cluster in children array
      final int[] newClusterIndices = new int[numberOfReferencePoints];
      // reference point for each index in children array
      final int[] clusterReferencePoints = new int[numberOfReferencePoints];
      
      // determine how many objects will be in each cluster
      for(DoubleDBIDListIter objectIterator = objects.iter(); objectIterator.valid(); objectIterator.advance()) {
        final int objectIndex = indexMap.get(DBIDUtil.deref(objectIterator));
        objectIDs[objectIterator.getOffset()] = objectIndex;
        final int nextReferencePoint = referencePointIDs[objectIndex][level + 1];
        if(newClusterSizes[nextReferencePoint] == 0) {
          newClusterIndices[nextReferencePoint] = numberOfNewClusters;
          clusterReferencePoints[numberOfNewClusters] = nextReferencePoint;
          numberOfNewClusters++;
        }
        newClusterSizes[referencePointIDs[objectIndex][level + 1]]++;
      }

      // create a list for each partition
      final ModifiableDoubleDBIDList[] partitionedObjects = new ModifiableDoubleDBIDList[numberOfNewClusters];
      final double[] minRadii = new double[numberOfNewClusters];
      final double[] maxRadii = new double[numberOfNewClusters];
      
      for(int i = 0; i < numberOfNewClusters; i++) {
        partitionedObjects[i] = DBIDUtil.newDistanceDBIDList(newClusterSizes[clusterReferencePoints[i]]);
        minRadii[i] = Double.POSITIVE_INFINITY;
        maxRadii[i] = Double.NEGATIVE_INFINITY;
      }

      // split objects into partitions
      final DoubleDBIDListIter objectIterator = objects.iter();
      for(int i = 0; i < size; i++) {
        partitionedObjects[newClusterIndices[referencePointIDs[objectIDs[i]][level + 1]]].add(objectIterator.doubleValue(), objectIterator);
        if(objectIterator.doubleValue() > maxRadii[newClusterIndices[referencePointIDs[i][level + 1]]])
          maxRadii[newClusterIndices[referencePointIDs[i][level + 1]]] = objectIterator.doubleValue();
        if(objectIterator.doubleValue() < minRadii[newClusterIndices[referencePointIDs[i][level + 1]]])
          minRadii[newClusterIndices[referencePointIDs[i][level + 1]]] = objectIterator.doubleValue();
        objectIterator.advance();
      }

      // sort partitions
      for(int i = 0; i < numberOfNewClusters; i++) {
        partitionedObjects[i].sort();
      }

      // create child nodes
      final TreeNode[] children = new TreeNode[numberOfNewClusters];
      for(int i = 0; i < numberOfNewClusters; i++) {
        children[i] = new LeafNode(clusterReferencePoints[i],primaryReferencePointIndex, level + 1, minRadii[i], maxRadii[i], partitionedObjects[i]);
      }

      // return new internal node with new children
      return new InternalNode(currentReferencePointIndex, size, level, rmin, rmax, children);
    }
  }

  /**
   * kNN query implementation.
   * 
   * @author Lennart Nöller
   */
  protected class DynamicMIndexKNNSearcher extends AbstractRefiningIndex<O>.AbstractRefiningQuery implements KNNSearcher<O> {
    /**
     * Constructor.
     * 
     * @param distanceQuery Distance query
     */
    public DynamicMIndexKNNSearcher(DistanceQuery<O> distanceQuery) {
      super(distanceQuery);
    }

    @Override
    public KNNList getKNN(O queryObject, int k) {
      // Array used for Pivot Filtering
      DoubleIntPair[] referencePointDistances = computeDistanceToReferencePoints(distanceQuery, queryObject, referencePoints);
      // Array used for iterating over clusters
      DoubleIntPair[] rankedReferencePoints = Arrays.copyOf(referencePointDistances, numberOfReferencePoints);
      Arrays.sort(rankedReferencePoints);

      int closestReferencePointID = rankedReferencePoints[0].second;

      KNNHeap kNNHeap = DBIDUtil.newHeap(k);
      double kDistanceUpperBound = Double.POSITIVE_INFINITY;
      for(DoubleIntPair currentPair : rankedReferencePoints) {
        final int pivot = currentPair.second;
        final double queryPivotDistance = currentPair.first;

        // Double Pivot Distance Constraint
        if((queryPivotDistance - rankedReferencePoints[0].first) > 2 * kDistanceUpperBound)
          break;
        final boolean[] inDoublePivotDistanceConstraint = doublePivotDistanceConstraint(referencePointDistances, closestReferencePointID, kDistanceUpperBound);
        recursiveKNNSearch(dynamicClusterTree[pivot], referencePointDistances, inDoublePivotDistanceConstraint, queryObject, closestReferencePointID, false, kNNHeap);

      }
      KNNList list = kNNHeap.toKNNList();
      return list;

    }

    /**
     * Find up to k nearest neighbors of the query object
     * 
     * @param root                            Root node of the current cluster
     *                                        tree
     * @param referencePointDistances         query-reference point distances
     * @param inDoublePivotDistanceConstraint if the double pivot distance
     *                                        constraint holds for a particular
     *                                        cluster
     * @param queryObject                     Query object
     * @param closestReferencePointID         ID of the reference point closest
     *                                        to the query object
     * @param wasInClosestCluster             if the current cluster is in the
     *                                        cluster of the query object
     * @param kNNHeap                         Heap that stores the found objects
     */
    @SuppressWarnings("unchecked")
    private void recursiveKNNSearch(TreeNode root, DoubleIntPair[] referencePointDistances, boolean[] inDoublePivotDistanceConstraint, O queryObject, int closestReferencePointID, boolean wasInClosestCluster, KNNHeap kNNHeap) {
      if(!wasInClosestCluster && !inDoublePivotDistanceConstraint[root.currentReferencePointIndex])
        return;
      double searchRadius = kNNHeap.getKNNDistance();
      if(root instanceof InMemoryDynamicMIndex.LeafNode) {
        LeafNode leaf = (LeafNode) root;
        final int pivot = leaf.currentReferencePointIndex;
        
        double lowerDistanceBound = referencePointDistances[pivot].first - searchRadius;
        double upperDistanceBound = referencePointDistances[pivot].first + searchRadius;

        // Range Pivot Distance Constraint
        if(upperDistanceBound < leaf.rmin)
          return;
        if(lowerDistanceBound > leaf.rmax)
          return;

        final ModifiableDoubleDBIDList currentPartition = leaf.objects;
        final DoubleDBIDListIter forwardIterator = currentPartition.iter(),
            backwardIterator = currentPartition.iter();

        // move iterator to first object with a larger distance to its
        // pivot than kDistanceUpperBound
        double queryPivotDistance = referencePointDistances[leaf.primaryReferencePointIndex].first;
        binarySearch(currentPartition, forwardIterator, queryPivotDistance);
        backwardIterator.seek(forwardIterator.getOffset() - 1);
        
        lowerDistanceBound = referencePointDistances[leaf.primaryReferencePointIndex].first - searchRadius;
        upperDistanceBound = referencePointDistances[leaf.primaryReferencePointIndex].first + searchRadius;
        
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
          if(maxValue <= searchRadius) {
            final double distance = refine(nextObject, queryObject);
            if(distance < searchRadius) {
              kNNHeap.insert(distance, nextObject);
              searchRadius = kNNHeap.getKNNDistance();
              
              //adjust bounds
              lowerDistanceBound = queryPivotDistance - searchRadius;
              upperDistanceBound = queryPivotDistance + searchRadius;
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
      else if(root instanceof InMemoryDynamicMIndex.InternalNode) {
        InternalNode node = (InternalNode) root;
        final boolean isClosest = node.currentReferencePointIndex == closestReferencePointID;
        Integer[] searchOrder = new Integer[node.children.length];
        for(int i = 0; i < searchOrder.length; i++) {
          searchOrder[i] = i;
        }
        Arrays.sort(searchOrder, new Comparator<Integer>() {
          @Override
          public int compare(Integer i1, Integer i2) {
              return Double.compare(referencePointDistances[node.children[i1].currentReferencePointIndex].first, referencePointDistances[node.children[i1].currentReferencePointIndex].first);
          }
      });
        for(int i = 0; i < node.children.length; i++) {
          recursiveKNNSearch(node.children[searchOrder[i]], referencePointDistances, inDoublePivotDistanceConstraint, queryObject, closestReferencePointID, isClosest, kNNHeap);
        }
      }
    }

  }

  /**
   * Exact range query implementation.
   * 
   * @author Lennart Nöller
   */
  protected class DynamicMIndexRangeSearcher extends AbstractRefiningIndex<O>.AbstractRefiningQuery implements RangeSearcher<O> {
    /**
     * Constructor.
     * 
     * @param distanceQuery Distance query
     */
    public DynamicMIndexRangeSearcher(DistanceQuery<O> distanceQuery) {
      super(distanceQuery);
    }

    @Override
    public ModifiableDoubleDBIDList getRange(O queryObject, double searchRadius, ModifiableDoubleDBIDList result) {
      // Array used for Pivot Filtering
      DoubleIntPair[] referencePointDistances = computeDistanceToReferencePoints(distanceQuery, queryObject, referencePoints);
      // Array used for iterating over clusters
      DoubleIntPair[] rankedReferencePoints = Arrays.copyOf(referencePointDistances, numberOfReferencePoints);
      Arrays.sort(rankedReferencePoints);

      int closestReferencePointID = rankedReferencePoints[0].second;

      final boolean[] inDoublePivotDistanceConstraint = doublePivotDistanceConstraint(referencePointDistances, closestReferencePointID, searchRadius);
      DBIDUtil.ensureArray(relation.getDBIDs());

      for(DoubleIntPair currentPair : rankedReferencePoints) {
        final int pivot = currentPair.second;
        final double queryPivotDistance = currentPair.first;

        // Double Pivot Distance Constraint
        // Since reference points are ordered we can stop when the condition is
        // first fulfilled
        if((queryPivotDistance - rankedReferencePoints[0].first) > 2 * searchRadius)
          break;

        // find all objects in current level 1 cluster within search radius
        final ModifiableDoubleDBIDList currentPartition = recusiveRangeSearch(dynamicClusterTree[pivot], referencePointDistances, inDoublePivotDistanceConstraint, queryObject, searchRadius, closestReferencePointID, pivot == closestReferencePointID);
        for(DoubleDBIDListIter partitionIterator = currentPartition.iter(); partitionIterator.valid(); partitionIterator.advance()) {
          result.add(partitionIterator.doubleValue(), partitionIterator);
        }
      }
      return result;
    }

    /**
     * Find all points within a cluster tree that are in range
     * 
     * @param root                            Root node of the current cluster
     *                                        tree
     * @param referencePointDistances         query-reference point distances
     * @param inDoublePivotDistanceConstraint if the double pivot distance
     *                                        constraint holds for a particular
     *                                        cluster
     * @param searchRadius                    query search radius
     * @param closestReferencePointID         ID of the reference point closest
     *                                        to
     *                                        q
     */
    @SuppressWarnings("unchecked")
    private ModifiableDoubleDBIDList recusiveRangeSearch(TreeNode root, DoubleIntPair[] referencePointDistances, boolean[] inDoublePivotDistanceConstraint, O queryObject, double searchRadius, int closestReferencePointID, boolean wasInClosestCluster) {
      if(!wasInClosestCluster && !inDoublePivotDistanceConstraint[root.currentReferencePointIndex])
        return DBIDUtil.newDistanceDBIDList();
      if(root instanceof InMemoryDynamicMIndex.LeafNode) {
        LeafNode leaf = (LeafNode) root;
        final int pivot = leaf.currentReferencePointIndex;
        double lowerDistanceBound = referencePointDistances[pivot].first - searchRadius;
        double upperDistanceBound = referencePointDistances[pivot].first + searchRadius;

        // Range Pivot Distance Constraint
        if(upperDistanceBound < leaf.rmin)
          return DBIDUtil.newDistanceDBIDList();
        if(lowerDistanceBound > leaf.rmax)
          return DBIDUtil.newDistanceDBIDList();

        final ModifiableDoubleDBIDList currentPartition = leaf.objects;
        final DoubleDBIDListIter dataPointIterator = currentPartition.iter();

        // move iterator to first object within distance range       
        lowerDistanceBound = referencePointDistances[leaf.primaryReferencePointIndex].first - searchRadius;
        upperDistanceBound = referencePointDistances[leaf.primaryReferencePointIndex].first + searchRadius;
        binarySearch(currentPartition, dataPointIterator, lowerDistanceBound);
        if(!dataPointIterator.valid() || dataPointIterator.doubleValue() > upperDistanceBound)
          return DBIDUtil.newDistanceDBIDList();
        final ModifiableDoubleDBIDList result = DBIDUtil.newDistanceDBIDList();

        // consider all objects within [d(q,pi) - r, d(q,pi) + r] range
        while(dataPointIterator.valid()) {
          double currentObjectDistance = dataPointIterator.doubleValue();
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
        }
        return result;
      }
      else if(root instanceof InMemoryDynamicMIndex.InternalNode) {
        InternalNode node = (InternalNode) root;
        final boolean isClosest = node.currentReferencePointIndex == closestReferencePointID;
        final ModifiableDoubleDBIDList result = DBIDUtil.newDistanceDBIDList();
        for(int i = 0; i < node.children.length; i++) {
          final ModifiableDoubleDBIDList childResult = recusiveRangeSearch(node.children[i], referencePointDistances, inDoublePivotDistanceConstraint, queryObject, searchRadius, closestReferencePointID, isClosest);
          if(childResult.isEmpty())
            continue;
          for(DoubleDBIDListIter resultIterator = childResult.iter(); resultIterator.valid(); resultIterator.advance()) {
            result.add(resultIterator.doubleValue(), resultIterator);
          }
        }
      }
      return DBIDUtil.newDistanceDBIDList();
    }
  }

  /**
   * Index factory for Dynamic M-Index indexes.
   * 
   * @author Lennart Nöller
   * 
   * @has - - - InMemoryDynamicMIndex
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
     * Max number of levels
     */
    int lmax;

    /**
     * Maximum number of elements in clusters on not max level
     */
    int maxClusterSize;

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
     * @param lmax                    Maximum number of levels
     * @param maxClusterSize          Maximum number of elements for non lmax
     *                                level clusters
     * @param numberOFStoredDistances Number of distances to be stored per data
     *                                point
     */
    public Factory(Distance<? super V> distance, KMedoidsInitialization<V> initialization, int k, int lmax, int maxClusterSize, int numberOfStoredDistances) {
      super();
      this.distance = distance;
      this.initialization = initialization;
      this.k = k;
      this.lmax = lmax;
      this.maxClusterSize = maxClusterSize;
      this.numberOfStoredDistances = numberOfStoredDistances;
    }

    @Override
    public InMemoryDynamicMIndex<V> instantiate(Relation<V> relation) {
      return new InMemoryDynamicMIndex<>(relation, distance.instantiate(relation), initialization, k, lmax, maxClusterSize, numberOfStoredDistances);
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
      public static final OptionID DISTANCE_ID = new OptionID("Dynamic M-Index.distance", "Distance function to build the index for.");

      /**
       * Initialization method.
       */
      public static final OptionID REFERENCE_ID = new OptionID("Dynamic M-Index.reference", "Method to choose the reference points.");

      /**
       * Number of reference points.
       */
      public static final OptionID K_ID = new OptionID("Dynamic M-Index.k", "Number of reference points to use.");

      /**
       * Number of distances stored per data point.
       */
      public static final OptionID STORED_DISTANCES_ID = new OptionID("Dynamic M-Index.stored_distances", "Number of distances stored per data point.");

      /**
       * Max number of layers.
       */
      public static final OptionID LMAX_ID = new OptionID("Dynamic M-Index.lmax", "Maximum number of layers.");

      /**
       * Maximum number of objects in non max level clusters.
       */
      public static final OptionID MAX_CLUSTER_SIZE_ID = new OptionID("Dynamix M-Index.maxClusterSize", "Maximum number of objects in non max level clusters.");

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
       * Number of distances stored
       */
      int numberOfStoredDistances;

      /**
       * Maximum number of levels
       */
      int lmax;

      /**
       * Maximum number of objects for non max levels clusters
       */
      int maxClusterSize;

      @Override
      public void configure(Parameterization config) {
        new ObjectParameter<Distance<? super V>>(DISTANCE_ID, Distance.class) //
            .grab(config, x -> distance = x);
        new ObjectParameter<KMedoidsInitialization<V>>(REFERENCE_ID, KMedoidsInitialization.class) //
            .grab(config, x -> initialization = x);
        new IntParameter(K_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> k = x);
        new IntParameter(LMAX_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> lmax = x);
        new IntParameter(MAX_CLUSTER_SIZE_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> maxClusterSize = x);
        new IntParameter(STORED_DISTANCES_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> numberOfStoredDistances = x);

      }

      @Override
      public InMemoryDynamicMIndex.Factory<V> make() {
        return new InMemoryDynamicMIndex.Factory<>(distance, initialization, k, lmax, maxClusterSize, numberOfStoredDistances);
      }
    }
  }
}
