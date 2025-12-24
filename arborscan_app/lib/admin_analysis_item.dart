class AdminAnalysisItem {
  final String analysisId;
  final String? species;
  final String? riskCategory;
  final double trustScore;

  AdminAnalysisItem({
    required this.analysisId,
    this.species,
    this.riskCategory,
    required this.trustScore,
  });

  factory AdminAnalysisItem.fromJson(Map<String, dynamic> json) {
    return AdminAnalysisItem(
      analysisId: json['analysis_id'],
      species: json['species'],
      riskCategory: json['risk_category'],
      trustScore: (json['trust_score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
