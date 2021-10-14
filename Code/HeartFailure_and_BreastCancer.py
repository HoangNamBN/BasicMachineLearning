import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def HeartFailure_and_BreastCancer(file_csv, classification_properties, properties):
    df = pd.read_csv(file_csv)
    print("Hiển thị 5 mẫu dữ liệu của file: \n", df.head())
    X = df.drop([classification_properties], axis=1)
    y = df[classification_properties]
    print("Số lượng nhãn của các lớp:\n", y.value_counts())
    print("Dữ liệu X: \n", X)
    print("Nhãn Y: \n", y)

    '''Chuẩn hoá'''
    std = StandardScaler()
    X = std.fit_transform(X)
    print("X sau khi được chuẩn hoá:\n", X)

    '''vẽ dữ liệu sử dụng PCA lên không gian 3 chiều'''
    print("Dữ liệu trước khi sử dụng PCA: ", X.shape)
    X = PCA(3).fit_transform(X)
    x_pca = X[:, 0]
    y_pca = X[:, 1]
    z_pca = X[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_pca, y_pca, z_pca, c=y, s=60)
    ax.legend(['Malign'])
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.show()
    print("Dữ liệu sau kho giảm chiều: ", X.shape)

    '''Biểu diễn mối quan hệ giữa các thành phần chính'''
    sns.scatterplot(x=x_pca, y=z_pca, hue=y, palette='Set1')
    plt.xlabel('First Principal Component')
    plt.ylabel('Third Principal Component')
    plt.show()

    sns.scatterplot(x=y_pca, y=z_pca, hue=y, palette='Set1')
    plt.xlabel('Second Principal Component')
    plt.ylabel('Third Principal Component')
    plt.show()

    sns.scatterplot(x=x_pca, y=y_pca, hue=y, palette='Set1')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

    '''Chia tập dữ liệu theo tỉ lệ 8:2'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
    print("Số dữ liệu train", len(X_train))
    print("Số dữ liệu test", len(X_test))
    print("Dữ liệu để train:\n", X_train)
    print("Nhãn dùng để train:\n", y_train)
    print("Dữ liệu dùng để test:\n", X_test)
    print("Hiển thị nhãn để test:\n", y_test)

    '''Chạy mô hình học máy: huấn luyện mô hình'''
    models = SVC(kernel='linear', C= 1).fit(X_train, y_train)

    '''Dự đoán mô hình'''
    y_predict = models.predict(X_test)
    print("Kết quả dự đoán:\n", y_predict)
    print("Hệ số w: ", models.coef_)
    print(models.coef_.shape)
    print("Hệ số bias: ", models.intercept_)
    print("Số lớp: ", models.classes_)

    '''Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)'''
    confusion_matrix1 = confusion_matrix(y_test, y_predict)
    print("Ma trận dự doán:\n", confusion_matrix1)
    print("Accuracy Score: ", accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict))

    X = df[[properties]]
    y = df[classification_properties]
    x0 = X[y == 0]
    x1 = X[y == 1]
    plt.plot(x0[properties], 'r^', markersize=4, alpha=.8)
    plt.plot(x1[properties], 'b^', markersize=4, alpha=.8)
    plt.xlabel(classification_properties)
    plt.ylabel(properties)
    plt.plot()
    plt.show()


if __name__ == "__main__":
    n = str(input("Nhập vào số nguyên để chọn chạy với dữ liệu: "))
    if n.isdigit() == True:
        print("Bạn đang chạy với bộ dữ liệu heart_failure_clinical_records_dataset.csv")
        HeartFailure_and_BreastCancer("heart_failure_clinical_records_dataset.csv", "DEATH_EVENT",
                                      "creatinine_phosphokinase")
    else:
        print("Bạn đang chạy với bộ dữ liệu Breast_cancer_data.csv")
        HeartFailure_and_BreastCancer("Breast_cancer_data.csv", "diagnosis", "mean_perimeter")