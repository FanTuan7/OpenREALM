#include <realm_maths/ransac.h>

namespace realm
{

RANSAC::RANSAC()
{
}

void RANSAC::setData(std::vector<MapPoint> &dataset_1, std::vector<MapPoint> &dataset_2)
{
    _in_dataset_1 = dataset_1;
    _in_dataset_2 = dataset_2;
    _index_all.resize(_in_dataset_1.size());
    for (int i = 0; i < _index_all.size(); i++)
    {
        _index_all[i] = i;
    }
}

void RANSAC::optimal_ransac(double general_tolerance, double final_tolerance)
{
    

    int flag = 0;
    int counter = 0;
    int num_of_data = _in_dataset_1.size();


    cv::Mat R, t;

    std::vector<int> inliers_quote;
    int num_quote = 0;

    std::vector<int> inliers;
    int num = 0;

    double bestErr = 999999999999;

    while (flag < 1 && counter <10000)
    {
        counter++;
        
        std::vector<int> samples;
        rand_sample(num_of_data, 6, samples);
        model(samples, R, t);
        std::vector<double> errors;
        score(_index_all, R, t, general_tolerance, inliers_quote, errors);
        num_quote = inliers_quote.size();

        if (num_quote > 7)
        {

            resample(R, t, final_tolerance, inliers_quote);

            num = inliers_quote.size();

            if (general_tolerance < final_tolerance)
            {

                pruneset(inliers_quote, R, t, general_tolerance);
                num_quote = inliers_quote.size();
            }
        }

        if ((num > 7) && (num == num_quote))
        {

            if (inliers.size() == inliers_quote.size())
            {

                flag++;
            }
            else
            {

                flag = 0;
                inliers = inliers_quote;
            }
        }
        else if (num_quote > num)
        {

            flag = 0;
            num = num_quote;
            inliers = inliers_quote;
        }
        else if (num_quote == (num - 1))
        {

            flag = 0;
            num = num_quote;
            inliers = inliers_quote;
        }
    }

    _out_inliers = inliers;
    _out_num_of_inliers = num;
    _out_R = R;
    _out_t = t;
}

/*
  https://en.wikipedia.org/wiki/Random_sample_consensus
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data
  */
void RANSAC::better_ransac(int n, int k, double threshold1, double threshold2, int d)
{
    int counter = 0;
    double bestErr = 999999999999;
    int num_of_data = _in_dataset_1.size();

    cv::Mat R, t;

    std::vector<int> inliers;
    int num = 0;

    int best_num = 0;
    std::vector<int> best_inliers;
    cv::Mat best_R, best_t;

    while (counter < k)
    {
        std::vector<int> samples;
        rand_sample(num_of_data, n, samples);
        model(samples, R, t);
        std::vector<double> errors;
        score(_index_all, R, t, threshold1, inliers, errors);
        num = inliers.size();

        if (num > 7)
        {
            resample(R, t, threshold1, inliers);
            pruneset(inliers, R, t, threshold2);
            num = inliers.size();
        }
        
        if (num > d)
        {
            double model_error = 0.0;
            for (int i = 0; i < inliers.size(); i++)
            {
                cv::Mat geo_pos1 = _in_dataset_1[inliers[i]]._mGeoPos;
                cv::Mat geo_pos2 = _in_dataset_2[inliers[i]]._mGeoPos;
                cv::Mat new_pos = R * geo_pos2 + t;
                model_error += sqrt(pow(geo_pos1.at<double>(0, 0) - new_pos.at<double>(0, 0), 2) + pow(geo_pos1.at<double>(1, 0) - new_pos.at<double>(1, 0), 2) + pow(geo_pos1.at<double>(2, 0) - new_pos.at<double>(2, 0), 2));
            }
            model_error = model_error / inliers.size();
            
            if ( (model_error < 2) && (model_error < bestErr))
            {
                bestErr = model_error;
                _out_num_of_inliers = num;
                _out_inliers = inliers;
                _out_R = R;
                _out_t = t;
                _out_error = bestErr;
            }
        }
        /*if (num > d)
        {
            if (num > best_num)
            {
                best_num = num;
                best_inliers = inliers;
                best_R = R;
                best_t = t;
            }
        }*/
        counter++;
    }
LOG_S(WARNING) << "     bestErr    :  " << bestErr;
   /* _out_inliers = best_inliers;
    _out_num_of_inliers = best_num;
    _out_R = best_R;
    _out_t = best_t;*/
}
/*
  https://en.wikipedia.org/wiki/Random_sample_consensus
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data
  */
void RANSAC::normal_ransac(int n, int k, double threshold, int d)
{
    int counter = 0;
    double bestErr = 999999999999;
    int num_of_data = _in_dataset_1.size();

    cv::Mat R, t;

    std::vector<int> inliers;
    int num = 0;

    int best_num = 0;
    std::vector<int> best_inliers;
    cv::Mat best_R, best_t;

    while (counter < k)
    {
        std::vector<int> samples;
        rand_sample(num_of_data, n, samples);
        model(samples, R, t);
        std::vector<double> errors;
        score(_index_all, R, t, threshold, inliers, errors);
        num = inliers.size();
        /*if (num > d)
        {
            if (num > best_num)
            {
                best_num = num;
                best_inliers = inliers;
                best_R = R;
                best_t = t;
            }
            _out_inliers = best_inliers;
    _out_num_of_inliers = best_num;
    _out_R = best_R;
    _out_t = best_t;
        }*/
        if (num > d)
        {   
            if (num > best_num)
            {
                best_num = num;
                best_inliers = inliers;
                best_R = R;
                best_t = t;

                double model_error = 0.0;
            for (int i = 0; i < inliers.size(); i++)
            {
                cv::Mat geo_pos1 = _in_dataset_1[inliers[i]]._mGeoPos;
                cv::Mat geo_pos2 = _in_dataset_2[inliers[i]]._mGeoPos;
                cv::Mat new_pos = R * geo_pos2 + t;
                model_error += sqrt(pow(geo_pos1.at<double>(0, 0) - new_pos.at<double>(0, 0), 2) + pow(geo_pos1.at<double>(1, 0) - new_pos.at<double>(1, 0), 2) + pow(geo_pos1.at<double>(2, 0) - new_pos.at<double>(2, 0), 2));
            }
            model_error = model_error / inliers.size();
             bestErr = model_error;
            }


               
                _out_num_of_inliers = best_num;
                _out_inliers = best_inliers;
                _out_R = best_R;
                _out_t = best_t;
                _out_error = bestErr;

            /*
            double model_error = 0.0;
            for (int i = 0; i < inliers.size(); i++)
            {
                cv::Mat geo_pos1 = _in_dataset_1[inliers[i]]._mGeoPos;
                cv::Mat geo_pos2 = _in_dataset_2[inliers[i]]._mGeoPos;
                cv::Mat new_pos = R * geo_pos2 + t;
                model_error += sqrt(pow(geo_pos1.at<double>(0, 0) - new_pos.at<double>(0, 0), 2) + pow(geo_pos1.at<double>(1, 0) - new_pos.at<double>(1, 0), 2) + pow(geo_pos1.at<double>(2, 0) - new_pos.at<double>(2, 0), 2));
            }
            model_error = model_error / inliers.size();
            
            if ( (model_error < 2) && (model_error < bestErr))
            {
                bestErr = model_error;
                _out_num_of_inliers = num;
                _out_inliers = inliers;
                _out_R = R;
                _out_t = t;
                _out_error = bestErr;
            }*/
        }
        counter++;
    }
LOG_S(WARNING) << "     _out_error    :  " << _out_error;
LOG_S(WARNING) << "     _out_num_of_inliers    :  " << _out_num_of_inliers;
    
 
}

void RANSAC::getResult(std::vector<int> &out_inliers, int &out_num_of_inliers, cv::Mat &out_R, cv::Mat &out_t, double &out_error)
{
    out_inliers = _out_inliers;
    out_num_of_inliers = _out_num_of_inliers;
    out_R = _out_R;
    out_t = _out_t;
    out_error = _out_error;
}

void RANSAC::rand_sample(int n, int m, std::vector<int> &samples)
{
    std::vector<int> random_num(n);
    std::vector<int> rand_samples(m);

    for (int i = 0; i < n; ++i)
    {
        random_num[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(random_num.begin(), random_num.end(), g);

    for (int i = 0; i < m; ++i)
    {
        rand_samples[i] = random_num[i];
    }

    samples = rand_samples;
}

void RANSAC::model(std::vector<int> &index, cv::Mat &maybe_R, cv::Mat &maybe_t)
{

    int size = index.size();

    std::vector<MapPoint> mps_1(size), mps_2(size);
    std::vector<cv::Point3f> pt3f_1, pt3f_2;

    for (int i = 0; i < size; i++)
    {
        mps_1[i] = _in_dataset_1[index[i]];
        mps_2[i] = _in_dataset_2[index[i]];
    }

    create_Point3f_from_MapPoint(mps_1, pt3f_1);
    create_Point3f_from_MapPoint(mps_2, pt3f_2);

    pose_estimation_3d3d(pt3f_1, pt3f_2, maybe_R, maybe_t);
}

void RANSAC::score(std::vector<int> &index, cv::Mat &R, cv::Mat &t, double tolerance, std::vector<int> &inliers, std::vector<double> &dist)
{
    std::vector<double> errors;
    double error = 0.0;
    std::vector<int> inliers_quote;

    for (int i = 0; i < index.size(); i++)
    {
        MapPoint mp1 = _in_dataset_1[index[i]];
        MapPoint mp2 = _in_dataset_2[index[i]];

        cv::Mat geo_pos1 = mp1._mGeoPos;
        cv::Mat geo_pos2 = mp2._mGeoPos;
        cv::Mat new_pos = R * geo_pos2 + t;

        error = sqrt(pow(geo_pos1.at<double>(0, 0) - new_pos.at<double>(0, 0), 2) + pow(geo_pos1.at<double>(1, 0) - new_pos.at<double>(1, 0), 2) + pow(geo_pos1.at<double>(2, 0) - new_pos.at<double>(2, 0), 2));

        if (error < tolerance)
        {
            inliers_quote.push_back(index[i]); //造成pruneset和score判断的点不一致。 不对，但确实是一处错误。
            errors.push_back(error);
        }
    }

    inliers = inliers_quote;
    dist = errors;
}

void RANSAC::resample(cv::Mat &R, cv::Mat &t, double tolerance, std::vector<int> &inliers)
{
    int num = inliers.size();
    int num_quote;
    int i = 0;
    std::vector<int> samples;
    std::vector<int> inliers_quote;
    cv::Mat R_quote, t_quote;

    while (i < 8)
    {
        i++;
        //int x = (int)std::round(num/2);
        rand_sample(num, std::max(6, (int)std::round(num/2)), samples);

        for (int j = 0; j < samples.size(); j++)
        {
            samples[j] = inliers[samples[j]];
        }

        model(samples, R_quote, t_quote);
        std::vector<double> errors;
        score(inliers, R_quote, t_quote, tolerance, inliers_quote, errors);
        num_quote = inliers_quote.size();

        if (num_quote > 7)
        {

            rescore(tolerance, inliers_quote, R_quote, t_quote, inliers_quote);
            int num_quote = inliers_quote.size();

            if (num_quote > num)
            {
                R = R_quote;
                t = t_quote;
                num = num_quote;
                inliers = inliers_quote;
                i = 0;
            }
        }
    }
}

void RANSAC::rescore(double in_tolerance, std::vector<int> inliers, cv::Mat &out_R, cv::Mat &out_t, std::vector<int> &out_inliers)
{
    int counter = 0;

    int num = inliers.size();

    int num_quote = inliers.size();
    std::vector<int> inliers_quote = inliers;

    while (counter < 20)
    {
        counter++;
        model(inliers, out_R, out_t);
        std::vector<double> errors;
        score(_index_all, out_R, out_t, in_tolerance, inliers, errors);
        num = inliers.size();
        if (num > 7)
        {
            if (num != num_quote)
            {
                num_quote = num;
                inliers_quote = inliers;
            }
            else
            {
                std::sort(inliers.begin(), inliers.end());
                std::sort(inliers_quote.begin(), inliers_quote.end());
                if (inliers == inliers_quote)
                {
                    counter = 20;
                }
                else
                {
                    num_quote = num;
                    inliers_quote = inliers;
                }
            }
        }
        else
        {
            counter = 20;
        }
    }
    out_inliers = inliers_quote;
}

void RANSAC::pruneset(std::vector<int> &inliers, cv::Mat &R, cv::Mat &t, double tolerance)
{
    bool flag = true;
    int max_dist_pos;
    double max_error;
    std::vector<double> errors;

    while (/*(inliers.size() > 7) && */ flag)
    {   


        score(inliers, R, t, tolerance, inliers, errors);

        max_error = 0;

        for (int i = 0; i < inliers.size(); i++)
        {
            if (errors[i] > max_error)
            {
                max_error = errors[i];
                max_dist_pos = i;
            }
        }
        //LOG_S(WARNING) << " max_error  :  " <<max_error;

        if (max_error > tolerance)
        {
            inliers.erase(inliers.begin() + max_dist_pos);
            model(inliers, R, t);
        }
        else
        {
            flag = false;
        }
    }
}

void RANSAC::create_Point3f_from_MapPoint(const std::vector<MapPoint> &mps, std::vector<cv::Point3f> &pts)
{
    double x, y, z;
    int n = mps.size();
    std::vector<cv::Point3f> tempo_pts(n);
    for(int i=0; i< n; i++)
    {
        x = mps[i]._mGeoPos.at<double>(0, 0);
        y = mps[i]._mGeoPos.at<double>(1, 0);
        z = mps[i]._mGeoPos.at<double>(2, 0);
        tempo_pts[i] = cv::Point3f(x, y, z);
    }

    pts = tempo_pts;
}

void RANSAC::create_Point3f_from_MapPoint_visual(const std::vector<MapPoint> &mps, std::vector<cv::Point3f> &pts)
{
    double x, y, z;
    int n = mps.size();
    std::vector<cv::Point3f> tempo_pts(n);
    for(int i=0; i< n; i++)
    {
        x = mps[i]._mWorldPos.at<double>(0, 0);
        y = mps[i]._mWorldPos.at<double>(1, 0);
        z = mps[i]._mWorldPos.at<double>(2, 0);
        tempo_pts[i] = cv::Point3f(x, y, z);
    }

    pts = tempo_pts;
}


void RANSAC::pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t)
{
    cv::Point3f p1, p2; // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    //LOG_S(WARNING) << "center of mass        p1:" << p1<<"             p2:"<<p2;
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // LOG_S(WARNING) << "q1:" << q1;
    //LOG_S(WARNING) << "q2:" << q2;
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    //LOG_S(WARNING) << "W:" << W;
    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    //LOG_S(WARNING) << "U:" << U;
    //LOG_S(WARNING) << "V:" << V;

    if (U.determinant() * V.determinant() < 0)
    {
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
    }

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2));
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

//helper for choose num_random elements from a vector
template <class bidiiter>
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random)
{
  size_t left = std::distance(begin, end);
  while (num_random--)
  {
    bidiiter r = begin;
    std::advance(r, rand() % left);
    std::swap(*begin, *r);
    ++begin;
    --left;
  }
  return begin;
}

} // namespace realm