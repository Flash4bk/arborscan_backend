import 'package:flutter/material.dart';

import 'admin_panel_page.dart';

/// Deprecated page kept only for backward compatibility.
///
/// Earlier builds of the app navigated to `AdminListPage`. The admin tooling has
/// been consolidated into `AdminPanelPage`, so this widget simply forwards.
class AdminListPage extends StatelessWidget {
  final String baseUrl;

  const AdminListPage({super.key, required this.baseUrl});

  @override
  Widget build(BuildContext context) {
    return AdminPanelPage(baseUrl: baseUrl);
  }
}
