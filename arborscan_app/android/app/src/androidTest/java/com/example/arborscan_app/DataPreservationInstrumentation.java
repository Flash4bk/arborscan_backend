package com.example.arborscan_app;

import android.app.Instrumentation;
import android.app.Activity;
import android.os.Bundle;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.security.MessageDigest;
import java.util.Arrays;

/** Installed only as a separately signed test APK; never part of ArborScan. */
public final class DataPreservationInstrumentation extends Instrumentation {
    @Override public void onCreate(Bundle arguments) { super.onCreate(arguments); start(); }
    private void collect(File root, File current, JSONObject inventory) throws Exception {
        if (!current.exists()) return;
        if (current.isDirectory()) {
            File[] children=current.listFiles();
            if (children == null) throw new IllegalStateException("Cannot inspect test data");
            Arrays.sort(children);
            for (File child: children) collect(root, child, inventory);
            return;
        }
        if (!current.getCanonicalPath().startsWith(root.getCanonicalPath()+File.separator))
            throw new IllegalStateException("Unexpected data path");
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try (FileInputStream input=new FileInputStream(current)) {
            byte[] bytes=new byte[65536]; int read;
            while ((read=input.read(bytes)) != -1) digest.update(bytes,0,read);
        }
        StringBuilder hex=new StringBuilder();
        for (byte b:digest.digest()) hex.append(String.format("%02x",b & 255));
        inventory.put(current.getAbsolutePath().substring(root.getAbsolutePath().length()+1),hex.toString());
    }
    @Override public void onStart() {
        Bundle result=new Bundle();
        try {
            File root=getTargetContext().getDataDir();
            JSONObject inventory=new JSONObject();
            for (String name:new String[]{"files","shared_prefs","app_flutter","no_backup"})
                collect(root,new File(root,name),inventory);
            result.putString("inventory",inventory.toString());
            result.putString("scope","Private SHA inventory only; no file contents or credentials");
            finish(Activity.RESULT_OK,result);
        } catch (Exception error) {
            result.putString("error","Private inventory could not be read");
            finish(Activity.RESULT_CANCELED,result);
        }
    }
}
