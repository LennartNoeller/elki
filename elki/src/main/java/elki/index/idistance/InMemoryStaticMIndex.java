package elki.index.idistance;

import elki.clustering.kmedoids.initialization.KMedoidsInitialization;
import elki.data.type.TypeInformation;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.IndexFactory;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

public class InMemoryStaticMIndex<O> extends InMemoryDynamicMIndex<O>{

  public InMemoryStaticMIndex(Relation<O> relation, DistanceQuery<O> distance, KMedoidsInitialization<O> initialization, int numberOfReferencePoints, int lmax, int numberOfStoredDistances) {
    super(relation, distance, initialization, numberOfReferencePoints, lmax, 0, numberOfStoredDistances);
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
    public Factory(Distance<? super V> distance, KMedoidsInitialization<V> initialization, int k, int lmax, int numberOfStoredDistances) {
      super();
      this.distance = distance;
      this.initialization = initialization;
      this.k = k;
      this.lmax = lmax;
      this.numberOfStoredDistances = numberOfStoredDistances;
    }

    @Override
    public InMemoryStaticMIndex<V> instantiate(Relation<V> relation) {
      return new InMemoryStaticMIndex<>(relation, distance.instantiate(relation), initialization, k, lmax, numberOfStoredDistances);
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
       * Max number of layers.
       */
      public static final OptionID LMAX_ID = new OptionID("Dynamic-M-Index.lmax", "Maximum number of layers.");

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
        new IntParameter(STORED_DISTANCES_ID)//
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> numberOfStoredDistances = x);

      }

      @Override
      public InMemoryStaticMIndex.Factory<V> make() {
        return new InMemoryStaticMIndex.Factory<>(distance, initialization, k, lmax, numberOfStoredDistances);
      }
    }
  }
}
